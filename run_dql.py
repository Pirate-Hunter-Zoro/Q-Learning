import sys

from frozen_lake import FrozenLakeDQL
from mountain_car import MountainCarDQL

if len(sys.argv) > 1:
    args = sys.argv[1:]

    if args[0] == 'frozen_lake':
        frozen_lake_dql = FrozenLakeDQL(is_slippery=True)
        frozen_lake_dql.train(episodes=10000)
        frozen_lake_dql.test(episodes=4)
    elif args[0] == 'mountain_car':
        mountain_car_dql = MountainCarDQL()
        mountain_car_dql.train(episodes=10000)
        mountain_car_dql.test(episodes=4)