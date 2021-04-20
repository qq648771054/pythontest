from lib import *
import threading
import envoriment

class ThreadBase(threading.Thread):
    def __init__(self, agentType, showProcess=True, savePath='', **kwargs):
        threading.Thread.__init__(self)
        self.agentType = agentType
        self.showProcess = showProcess
        self.savePath = savePath
        self.args = kwargs

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

class ThreadMaze(ThreadBase):
    WIDTH, HEIGHT = 5, 5
    def run(self):
        mapDatas = []
        mapIter = 0
        dir = getDataFilePath('DQN_Maze')
        for file in os.listdir(dir):
            file = os.path.join(dir, file)
            mapDatas.append(readFile(file))
        env = envoriment.Maze(self.WIDTH, self.WIDTH, self.agentType)
        agent = env.agent
        self.loadModel(agent)
        episode = 0
        record = [0] * 50
        success = 0
        while True:
            state = env.reset(mapDatas[mapIter])
            self.render(env)
            episode += 1
            startTime = time.time()
            step = 0
            while True:
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)
                agent.save_exp(state, action, reward, next_state)
                step += 1
                state = next_state
                self.render(env, 0.05)
                if done:
                    break
            isSuccess = int(reward == 1)
            print('episode {}, mapIter {}, result {}, takes {} steps {} second'.format(episode, mapIter, isSuccess, step, time.time() - startTime))
            success += isSuccess - record[episode % 50]
            record[episode % 50] = isSuccess
            agent.learn()
            self.saveModel(agent)
            if success >= 30:
                record = [0] * 50
                success = 0
                mapIter = (mapIter + 1) % len(mapDatas)
                print('switch mapIter to {}'.format(mapIter))

class ThreadCartPole(ThreadBase):
    def run(self):
        env = envoriment.CartPole_v0(self.agentType)
        agent = env.agent
        self.loadModel(agent)
        episode = 0
        while True:
            state = env.reset()
            self.render(env, 0.5)
            episode += 1
            step = 0
            while True:
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)
                agent.save_exp(state, action, reward, next_state, done)
                step += 1
                state = next_state
                self.render(env)
                if done:
                    break
            agent.learn()
            print('episode {}, steps {}'.format(episode, step))
            self.saveModel(agent)