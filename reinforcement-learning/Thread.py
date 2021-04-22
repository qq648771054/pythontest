from lib import *
import threading
import envoriment

class ThreadBase(threading.Thread):
    def __init__(self, agentType, showProcess=True, savePath='', **kwargs):
        threading.Thread.__init__(self)
        self.agentType = agentType
        self.showProcess = showProcess
        self.savePath = savePath
        for k, v in kwargs.items():
            setattr(self, k, v)

    def loadModel(self, agent):
        if self.savePath and os.path.exists(self.savePath):
            agent.model = tf.keras.models.load_model(self.savePath)

    def saveModel(self, agent):
        if self.savePath:
            agent.model.save(self.savePath)

    def render(self, env, sleepTime=None):
        if self.showProcess:
            env.render()
            if sleepTime:
                time.sleep(sleepTime)