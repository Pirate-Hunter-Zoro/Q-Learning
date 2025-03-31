import sys

from frozen_lake import FrozenLakeDQL
from mountain_car import MountainCarDQL
from car_racing import CarRacingDQL

if len(sys.argv) > 1:
    args = sys.argv[1:]

    if args[0] == 'frozen_lake':
        frozen_lake_dql = FrozenLakeDQL(is_slippery=True)
        frozen_lake_dql.train(episodes=10000)
    elif args[0] == 'mountain_car':
        mountain_car_dql = MountainCarDQL()
        mountain_car_dql.train(episodes=10000)
    elif args[0] == 'carracing':
        carracing_dql = CarRacingDQL()
        carracing_dql.train(episodes=600)