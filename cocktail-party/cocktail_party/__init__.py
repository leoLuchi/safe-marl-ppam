from gymnasium.envs.registration import register

register(
    id="cocktail_party-v0",
    entry_point="cocktail_party.envs:CPEnvMulti",
    max_episode_steps=500,
)
