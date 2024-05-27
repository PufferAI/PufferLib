import psutil
import GPUtil
import time
import sys

import rich
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn

#import pufferlib

ROUND_OPEN = rich.box.Box(
    "╭──╮\n"
    "│  │\n"
    "│  │\n"
    "│  │\n"
    "│  │\n"
    "│  │\n"
    "│  │\n"
    "╰──╯\n"
)

c1 = '[bright_cyan]'
c2 = '[white]'
c3 = '[cyan]'
b1 = '[bright_cyan]'
b2 = '[bright_white]'

def abbreviate(num):
    if num < 1e3:
        return f"{num:.0f}"
    elif num < 1e6:
        return f"{num/1e3:.1f}k"
    elif num < 1e9:
        return f"{num/1e6:.1f}m"
    elif num < 1e12:
        return f"{num/1e9:.1f}b"
    else:
        return f"{num/1e12:.1f}t"

def duration(seconds):
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}h {m}m {s}s" if h else f"{m}m {s}s" if m else f"{s}s"

def print_dashboard(performance_data, loss_data, user_data, min_interval=0.25, last_print=[0]):
    console = Console()

    util = Table(box=None, expand=True, show_header=False)
    cpu_percent = psutil.cpu_percent()
    dram_percent = psutil.virtual_memory().percent
    gpus = GPUtil.getGPUs()
    gpu_percent = gpus[0].load * 100 if gpus else 0
    vram_percent = gpus[0].memoryUtil * 100 if gpus else 0
    util.add_column(justify="left")
    util.add_column(justify="center")
    util.add_column(justify="center")
    util.add_column(justify="center")
    util.add_column(justify="right")
    util.add_row(
        f':blowfish: {c1}PufferLib {b2}1.0.0',
        f'{c1}CPU: {c3}{cpu_percent:.1f}%',
        f'{c1}GPU: {c3}{gpu_percent:.1f}%',
        f'{c1}DRAM: {c3}{dram_percent:.1f}%',
        f'{c1}VRAM: {c3}{vram_percent:.1f}%',
    )
        
    summary= Table(box=None, expand=True)
    summary.add_column(f"{c1}Summary", justify='left', vertical='top')
    summary.add_column(f"{c1}Value", justify='right', vertical='top')
    summary.add_row(f'{c2}Epoch', f'{b2}{performance.epoch}')
    summary.add_row(f'{c2}Uptime', f'{b2}{duration(performance.uptime)}')
    estimated_time = performance.total_steps / performance.sps
    summary.add_row(f'{c2}Estim', f'{b2}{duration(estimated_time)}')
    summary.add_row(f'{c2}Agent Steps', f'{b2}{abbreviate(performance.agent_steps)}')
    summary.add_row(f'{c2}Steps/sec', f'{b2}{abbreviate(performance.sps)}')
    summary.add_row(f'{c2}sec/Batch', f'{b2}{performance.epoch_time:.2f}')
   
    perf = Table(box=None, expand=True)
    perf.add_column(f"{c1}Performance", justify="left", ratio=1.0)
    perf.add_column(f"{c1}Time", justify="right", ratio=0.5)
    perf.add_row(f'{c2}Training', f'{b2}{performance.epoch_train_time:.2f}')
    perf.add_row(f'{c2}Evaluation', f'{b2}{performance.epoch_eval_time:.2f}')
    perf.add_row(f'{c2}Environment', f'{b2}{performance.epoch_env_time:.2f}')
    perf.add_row(f'{c2}Forward', f'{b2}{performance.epoch_forward_time:.2f}')
    perf.add_row(f'{c2}Misc', f'{b2}{performance.epoch_misc_time:.2f}')
    perf.add_row(f'{c2}Allocation', f'{b2}{performance.epoch_alloc_time:.2f}')
    perf.add_row(f'{c2}Backward', f'{b2}{performance.epoch_backward_time:.2f}')

    losses = Table(box=None, expand=True)
    losses.add_column(f'{c1}Losses', justify="left", ratio=1.0)
    losses.add_column(f'{c1}Value', justify="right", ratio=0.5)
    for metric, value in loss_data.items():
        losses.add_row(f'{c2}{metric}', f'{b2}{value}')

    monitor = Table(box=None, expand=True, pad_edge=False)
    monitor.add_row(summary, perf, losses)

    user = Table(box=None, expand=True, pad_edge=False)
    user1 = Table(box=None, expand=True)
    user2 = Table(box=None, expand=True)
    user.add_row(user1, user2)
    user1.add_column(f"{c1}User Stats", justify="left", ratio=1.0)
    user1.add_column(f"{c1}Value", justify="right",ratio=1.0)
    user2.add_column(f"{c1}User Stats", justify="left", ratio=1.0)
    user2.add_column(f"{c1}Value", justify="right",ratio=1.0)
    i = 0
    for metric, value in user_data.items():
        u = user1 if i % 2 == 0 else user2
        u.add_row(f'{c2}{metric}', f'{b2}{value}')
        i += 1

    table = Table(box=ROUND_OPEN, expand=True, show_header=False, width=80, border_style='bright_cyan')
    table.add_row(util)
    table.add_row(monitor)
    table.add_row(user)
    console.print(table)


class Dashboard:
    def __init__(self):
        self.console = Console()
        self.rich = rich

        layout = Layout()
        layout.split(
            Layout(name="utilization", size=5),
            Layout(name="monitoring"),
        )
 
        self.layout = layout
        '''
        layout.split(
            Layout(name="utilization", size=5),
            Layout(name="puffer", size=2),
            Layout(name="monitor", size=12),
            Layout(name="user")
        )
        layout["monitor"].split_row(
            Layout(name="performance"),
            Layout(name="losses")
        )
        layout["user"].split_row(
            Layout(name="user_stats")
        )
        '''

        utilization = Progress(
            BarColumn(bar_width=None, style="bar.back", complete_style="bar.complete"),
            TextColumn("[progress.description]{task.description}"),
            MofNCompleteColumn(),
            expand=True
        )
        self.cpu_task = utilization.add_task("[cyan]CPU", total=100)
        self.gpu_task = utilization.add_task("[red]GPU", total=100)
        self.dram_task = utilization.add_task("[blue]DRAM", total=100)
        self.vram_task = utilization.add_task("[magenta]VRAM", total=100)
        self.layout["utilization"].update(utilization)
        self.utilization = utilization

        #self.live = Live(self.layout, console=self.console)#, auto_refresh=4)
        #self.live.start()

    def _update_utilization(self):
        self.utilization.update(self.cpu_task, completed=psutil.cpu_percent())
        self.utilization.update(self.dram_task, completed=psutil.virtual_memory().percent)
        gpus = GPUtil.getGPUs()
        if gpus:
            self.utilization.update(self.gpu_task, completed=gpus[0].load * 100)
            self.utilization.update(self.vram_task, completed=gpus[0].memoryUtil * 100)
        else:
            self.utilization.update(self.gpu_task, completed=0)
            self.utilization.update(self.vram_task, completed=0)

        #self.layout['puffer'].update(f':blowfish: PufferLib {pufferlib.__version__}')
        #self.layout['puffer'].update(f':blowfish: PufferLib 1.0.0')

    def update(self, total_uptime, estimated_time, total_steps, steps_per_second, performance_data, loss_data, user_data):
        topline = self.update_topline(total_uptime, estimated_time, total_steps, steps_per_second)
        performance = self.update_performance(performance_data)
        losses = self.update_losses(loss_data)
        user = self.update_user_stats(user_data)

        megatable = Table(box=ROUND_OPEN, expand=True, show_header=False)
        megatable.add_row(topline)
        megatable.add_row('')
        perf = Table(box=None, expand=True)
        perf.add_column(performance, ratio=1.0)
        perf.add_column(losses, ratio=1.0)
        #megatable.add_row(performance)
        #megatable.add_row(losses)
        megatable.add_row(perf)
        megatable.add_row('')
        megatable.add_row(user)
        self.layout["monitoring"].update(megatable)
        self.console.clear()
        self.console.print(self.layout) 


    def update_topline(self, total_uptime, estimated_time, total_steps, steps_per_second):
        table = Table(box=None, expand=True)
        table.add_column(justify="left")
        table.add_column(justify="center")
        table.add_column(justify="right")
        table.add_row(
            f':blowfish: PufferLib 1.0.0',
            f'[bold magenta]Uptime: [cyan]{total_uptime}/{estimated_time}(est)',
            f'[bold magenta]Agent Steps: [cyan]{total_steps} ({steps_per_second}/s)'
        )
        return table

    def update_performance(self, data):
        table = Table(box=None, expand=True)
        #self.layout["performance"].update(table)
        table.add_column("[bold magenta]Performance", justify="right", ratio=1.0)
        table.add_column("Latency", justify="left", style="cyan", ratio=1.0)
        for metric, value in data.items():
            table.add_row(metric, str(value))

        return table
        self.console.clear()
        self.console.print(self.layout) 

    def update_losses(self, data):
        table = Table(box=None, expand=True)
        #self.layout["losses"].update(table)
        table.add_column("[bold magenta]Losses", justify="right", ratio=1.0)
        table.add_column("Value", justify="left", style="bright_cyan", ratio=1.0)
        for metric, value in data.items():
            table.add_row(metric, str(value))

        table.add_row("")

        return table
        self.console.clear()
        self.console.print(self.layout) 

    def update_user_stats(self, data):
        table = Table(box=None, expand=True)
        table.add_column("[bold magenta]User Stats", justify="right", style="bold yellow", ratio=1.0)
        table.add_column("Value", justify="left",ratio=1.0)
        #self.layout["user_stats"].update(table)
        for metric, value in data.items():
            table.add_row(metric, str(value))

        return table
        self.console.clear()
        self.console.print(self.layout) 


#dashboard = Dashboard()

# Update loop
try:
    while True:
        #dashboard._update_utilization()
        topline = (5000, 100000, 102332, 1038, 1.3)
        performance = {
            'training': 0.7,
            'evaluation': 0.6,
            'environment': 0.2,
            'forward': 0.3,
            'misc': 0.1,
            'allocation': 0.2,
            'backward': 0.3,
        }
        losses = {
            'policy': 0.4,
            'value': 0.2,
            'entropy': 0.1,
            'old_approx_kl': 0.1,
            'approx_kl': 0.2,
            'clip_fraction': 0.1,
            'explained_variance': 0.3,
        }
        user_stats = {
            'time_alive': 128,
            'exploration': 0.1,
            'experience': 1000,
        }
        #dashboard.update(*topline, performance, losses, user_stats)
        print_dashboard(*topline, performance, losses, user_stats)
        time.sleep(1)
except KeyboardInterrupt:
    dashboard.stop()

