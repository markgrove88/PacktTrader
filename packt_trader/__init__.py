from gym.envs.registration import register

register(
    id='packttrade-v0',
    entry_point='packt_trader.envs:TradingEnvironment',
    max_episode_steps=252)
)
