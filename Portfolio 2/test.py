import tmrl

env = tmrl.get_environment()
obs, info = env.reset()
print(type(obs))
print(obs)