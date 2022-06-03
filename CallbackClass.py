class CallbackClass():
    def __init__(self, strategy, labels):
        self.strategy = strategy
        for strat in self.strategy: 
            setattr(self, f'solution_{strat}', [])
            setattr(self, f'lscale_{strat}', [])
            setattr(self, f'ARI_{strat}', [])
            setattr(self, f'NMI_{strat}', [])
            setattr(self, f'inclusion_{strat}', [])
            setattr(self, f'objagree_{strat}', [])
            setattr(self, f'noise_{strat}', [])
            setattr(self, f'max_{strat}', [])
            setattr(self, f'labels_{strat}', labels)

    def append_all_tasks_to(self, listname, key_part, log_dict, labels):
        task_vals = [value for key,value in log_dict.items() if key.startswith(key_part)] 
        num_tasks = len(task_vals)
        ordered_task_vals = []
        for i in range(num_tasks):
            ordered_task_vals.append(log_dict[key_part + f"/{labels[i]}"])
        if num_tasks > 0:
            listname.append(ordered_task_vals)

    def append_to(self, listname, key, log_dict):
        if key in log_dict:
            listname.append(log_dict[key])

    def __call__(self, log_dict):
        for strat in self.strategy:
            self.append_to(getattr(self, f'solution_{strat}'), f"Solution/{strat}", log_dict)
            self.append_to(getattr(self, f'lscale_{strat}'), f"Lengthscale/{strat}", log_dict)
            self.append_to(getattr(self, f'ARI_{strat}'), f"ARI/{strat}", log_dict)
            self.append_to(getattr(self, f'NMI_{strat}'), f"NMI/{strat}", log_dict)
            self.append_all_tasks_to(getattr(self, f'inclusion_{strat}'), f"Inclusion probability/{strat}", log_dict, getattr(self, f'labels_{strat}'))
            self.append_all_tasks_to(getattr(self, f'objagree_{strat}'), f"Correlation/{strat}", log_dict, getattr(self, f'labels_{strat}'))
            self.append_all_tasks_to(getattr(self, f'noise_{strat}'), f"Noise/{strat}", log_dict, getattr(self, f'labels_{strat}'))
            self.append_all_tasks_to(getattr(self, f'max_{strat}'), f"Max/{strat}", log_dict, getattr(self, f'labels_{strat}'))
