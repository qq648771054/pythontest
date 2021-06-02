from lib import *
from OpenAiGym.GameBase import *
class MountainCarBase(GameBase):
    saveType = {
        'episode': (str, int, 0),
        'spendTime': (str, float, 0)
    }
    bakFrequence = 500

    actionLen = 3
    stateLen = 2

    def train(self, showProcess=False):
        pass

    def play(self):
        super(MountainCarBase, self).play()
        while True:
            self.env.reset()
            self.render()
            self.action = -1
            while True:
                self.render(0.016)
                if self.action != -1:
                    state, reward, done, info = self.env.step(self.action)
                    if done:
                        break

    def on_key_press(self, key, mode):
        if key == KEY.LEFT or key == KEY.A:
            self.action = 0
        elif key == KEY.RIGHT or key == KEY.D:
            self.action = 1

if __name__ == '__main__':
    game = MountainCarBase('MountainCar-v0')
    game.play()
