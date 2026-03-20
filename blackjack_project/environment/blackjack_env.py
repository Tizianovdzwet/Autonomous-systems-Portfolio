import gymnasium as gym

class BlackjackGymEnv:
    def __init__(self, render=False):
        # We gebruiken de standaard Gymnasium omgeving
        mode = "human" if render else None
        self.env = gym.make("Blackjack-v1", render_mode=mode)
    
    def reset(self):
        obs, info = self.env.reset()
        return obs
    
    def step(self, action):
        # Gymnasium Blackjack acties: 0 = Stick (Stand), 1 = Hit
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()