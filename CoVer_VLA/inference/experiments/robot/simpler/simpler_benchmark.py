from experiments.robot.simpler.simpler_utils import get_simpler_env

BENCHMARK_MAPPING = {}


def register_benchmark(target_class):
    """We design the mapping to be case-INsensitive."""
    # Create an instance to get the name attribute
    instance = target_class()
    BENCHMARK_MAPPING[instance.name.lower()] = target_class


def get_benchmark(benchmark_name):
    return BENCHMARK_MAPPING[benchmark_name.lower()]


###

task_map = {
    "simpler_widowx": [
        "widowx_put_eggplant_in_basket",
        "widowx_spoon_on_towel",
        "widowx_stack_cube",
        "widowx_carrot_on_plate",
    ],
    "simpler_stack_cube": [
        "widowx_stack_cube",
    ],
    "simpler_put_eggplant_in_basket": [
        "widowx_put_eggplant_in_basket",
    ],
    "simpler_spoon_on_towel": [
        "widowx_spoon_on_towel",
    ],
    "simpler_carrot_on_plate": [
        "widowx_carrot_on_plate",
    ],
    "simpler_redbull_on_plate": [
        "widowx_redbull_on_plate",
    ],
    "simpler_carrot_on_plate_unseen_lighting": [
        "widowx_carrot_on_plate_unseen_lighting",
    ],
    "simpler_tennis_ball_in_basket": [
        "widowx_tennis_ball_in_basket",
    ],
    "simpler_toy_dinosaur_on_towel": [
        "widowx_toy_dinosaur_on_towel",
    ],
    "simpler_zucchini_on_towel": [
        "widowx_zucchini_on_towel",
    ],
    "simpler_ood":[
        "widowx_redbull_on_plate",
        "widowx_zucchini_on_towel",
        "widowx_tennis_ball_in_basket",
        # "widowx_toy_dinosaur_on_towel",
    ]
}

class Benchmark:
    def _make_benchmark(self):
        self.tasks = task_map[self.name]

    def get_task(self, i):
        return self.tasks[i]

    def make(self, *args, **kwargs):
        return self.env_fn(*args, **kwargs)

    @property
    def n_tasks(self):
        return len(self.tasks)


class SimplerBenchmark(Benchmark):
    def __init__(self):
        super().__init__()
        self.env_fn = get_simpler_env
        self.state_dim = 7


@register_benchmark
class SIMPLER_WIDOWX(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_widowx"
        self._make_benchmark()


@register_benchmark
class SIMPLER_WIDOWX_CUBE(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_stack_cube"
        self._make_benchmark()

@register_benchmark
class SIMPLER_WIDOWX_EGGPLANT(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_put_eggplant_in_basket"
        self._make_benchmark()

@register_benchmark
class SIMPLER_WIDOWX_SPOON(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_spoon_on_towel"
        self._make_benchmark()

@register_benchmark
class SIMPLER_WIDOWX_CARROT(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_carrot_on_plate"
        self._make_benchmark()

@register_benchmark
class SIMPLER_WIDOWX_REDBULL(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_redbull_on_plate"
        self._make_benchmark()
        
        
@register_benchmark
class SIMPLER_WIDOWX_TENNIS_BALL(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_tennis_ball_in_basket"
        self._make_benchmark()
        
@register_benchmark
class SIMPLER_WIDOWX_CARROT_UNSEEN_LIGHTING(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_carrot_on_plate_unseen_lighting"
        self._make_benchmark()

@register_benchmark
class SIMPLER_WIDOWX_TOY_DINOSAUR(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_toy_dinosaur_on_towel"
        self._make_benchmark()
        
@register_benchmark
class SIMPLER_WIDOWX_ZUCCHINI(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_zucchini_on_towel"
        self._make_benchmark()

@register_benchmark
class SIMPLER_WIDOWX_OOD(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_ood"
        self._make_benchmark()