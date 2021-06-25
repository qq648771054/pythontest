from lib import *
from OpenAiGym.GameBase import *
class BreakOutBase(GameBase):
    saveType = {
        'episode': (str, int, 0),
        'totalStep': (str, int, 0),
        'spendTime': (str, float, 0),
    }
    bakFrequence = 500

    actionLen = 4
    stateSize = (128, )

    def train(self, showProcess=False):
        pass

    def play(self):
        super(BreakOutBase, self).play()
        while True:
            self.env.reset()
            self.render()
            while True:
                self.render(0.016)
                state, reward, done, info = self.env.step(self.env.action_space.sample())
                if done:
                     break
            print(f'reward {reward}')

    def on_key_press(self, key, mode):
        if key == KEY.LEFT or key == KEY.A:
            self.action = 0
        elif key == KEY.RIGHT or key == KEY.D:
            self.action = 1

if __name__ == '__main__':
    game = BreakOutBase('Breakout-ram-v0')
    game.play()
