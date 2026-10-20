"""Drain (He et al., ICWS 2017) -- streaming version of the logpai implementation.

The tree logic (treeSearch, addSeqToPrefixTree, seqDist, fastMatch, getTemplate) is
copied from logpai/logparser Drain (identical to the copy in Loghub-2.0 and in
Qin et al.'s artifact). The only change is the I/O: instead of loading the whole
file into a pandas DataFrame and keeping a list of line ids per cluster, this
class consumes one token list at a time and returns the cluster index, so the
11M-line HDFS file can be parsed in bounded memory. Final templates are
read from the clusters at the end, which is exactly what logpai's
``outputResult`` does (a line gets its cluster's *final* template).
"""


class Logcluster:
    __slots__ = ("logTemplate", "idx", "size")

    def __init__(self, logTemplate, idx):
        self.logTemplate = logTemplate
        self.idx = idx
        self.size = 0


class Node:
    __slots__ = ("childD", "depth", "digitOrtoken")

    def __init__(self, childD=None, depth=0, digitOrtoken=None):
        self.childD = dict() if childD is None else childD
        self.depth = depth
        self.digitOrtoken = digitOrtoken


class Drain:
    def __init__(self, depth=4, st=0.5, maxChild=100):
        self.depth = depth - 2
        self.st = st
        self.maxChild = maxChild
        self.rootNode = Node()
        self.clusters = []

    @staticmethod
    def hasNumbers(s):
        return any(char.isdigit() for char in s)

    def treeSearch(self, rn, seq):
        seqLen = len(seq)
        if seqLen not in rn.childD:
            return None
        parentn = rn.childD[seqLen]
        currentDepth = 1
        for token in seq:
            if currentDepth >= self.depth or currentDepth > seqLen:
                break
            if token in parentn.childD:
                parentn = parentn.childD[token]
            elif '<*>' in parentn.childD:
                parentn = parentn.childD['<*>']
            else:
                return None
            currentDepth += 1
        return self.fastMatch(parentn.childD, seq)

    def addSeqToPrefixTree(self, rn, logClust):
        seqLen = len(logClust.logTemplate)
        if seqLen not in rn.childD:
            firtLayerNode = Node(depth=1, digitOrtoken=seqLen)
            rn.childD[seqLen] = firtLayerNode
        else:
            firtLayerNode = rn.childD[seqLen]
        parentn = firtLayerNode
        currentDepth = 1
        for token in logClust.logTemplate:
            if currentDepth >= self.depth or currentDepth > seqLen:
                if len(parentn.childD) == 0:
                    parentn.childD = [logClust]
                else:
                    parentn.childD.append(logClust)
                break
            if token not in parentn.childD:
                if not self.hasNumbers(token):
                    if '<*>' in parentn.childD:
                        if len(parentn.childD) < self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken=token)
                            parentn.childD[token] = newNode
                            parentn = newNode
                        else:
                            parentn = parentn.childD['<*>']
                    else:
                        if len(parentn.childD) + 1 < self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken=token)
                            parentn.childD[token] = newNode
                            parentn = newNode
                        elif len(parentn.childD) + 1 == self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken='<*>')
                            parentn.childD['<*>'] = newNode
                            parentn = newNode
                        else:
                            parentn = parentn.childD['<*>']
                else:
                    if '<*>' not in parentn.childD:
                        newNode = Node(depth=currentDepth + 1, digitOrtoken='<*>')
                        parentn.childD['<*>'] = newNode
                        parentn = newNode
                    else:
                        parentn = parentn.childD['<*>']
            else:
                parentn = parentn.childD[token]
            currentDepth += 1

    @staticmethod
    def seqDist(seq1, seq2):
        simTokens = 0
        numOfPar = 0
        for token1, token2 in zip(seq1, seq2):
            if token1 == '<*>':
                numOfPar += 1
                continue
            if token1 == token2:
                simTokens += 1
        return float(simTokens) / len(seq1), numOfPar

    def fastMatch(self, logClustL, seq):
        maxSim = -1
        maxNumOfPara = -1
        maxClust = None
        for logClust in logClustL:
            curSim, curNumOfPara = self.seqDist(logClust.logTemplate, seq)
            if curSim > maxSim or (curSim == maxSim and curNumOfPara > maxNumOfPara):
                maxSim = curSim
                maxNumOfPara = curNumOfPara
                maxClust = logClust
        if maxSim >= self.st:
            return maxClust
        return None

    @staticmethod
    def getTemplate(seq1, seq2):
        return [w if w == seq2[i] else '<*>' for i, w in enumerate(seq1)]

    def add(self, logmessageL):
        """Insert one tokenised (pre-processed) message, return its cluster index."""
        matchCluster = self.treeSearch(self.rootNode, logmessageL)
        if matchCluster is None:
            matchCluster = Logcluster(logTemplate=logmessageL, idx=len(self.clusters))
            self.clusters.append(matchCluster)
            self.addSeqToPrefixTree(self.rootNode, matchCluster)
        else:
            newTemplate = self.getTemplate(logmessageL, matchCluster.logTemplate)
            if newTemplate != matchCluster.logTemplate:
                matchCluster.logTemplate = newTemplate
        matchCluster.size += 1
        return matchCluster.idx

    def templates(self):
        return [' '.join(c.logTemplate) for c in self.clusters]
