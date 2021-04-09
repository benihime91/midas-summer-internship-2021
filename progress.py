"""
Fancy progress bar callback for PyTorch Lightning.
This was inspired from: https://github.com/huggingface/transformers/blob/master/src/transformers/utils/notebook.py
"""
import time
from typing import *

import IPython.display as disp
import torch
from pytorch_lightning.callbacks import ProgressBarBase


##################################################################################################################
# Straight copy from : https://github.com/huggingface/transformers/blob/master/src/transformers/utils/notebook.py
###################################################################################################################

def format_time(t):
    "Format `t` (in seconds) to (h):mm:ss"
    t = int(t)
    h, m, s = t // 3600, (t // 60) % 60, t % 60
    return f"{h}:{m:02d}:{s:02d}" if h != 0 else f"{m:02d}:{s:02d}"


def html_progress_bar(value, total, prefix, label, width=300):
    # docstyle-ignore
    return f"""
    <div>
        <style>
            progress {{
                border: none;
                background-size: auto;
            }}
        </style>
      {prefix}
      <progress value='{value}' max='{total}' style='width:{width}px; height:20px; vertical-align: middle;'></progress>
      {label}
    </div>
    """


def text_to_html_table(items):
    "Put the texts in `items` in an HTML table."
    html_code = """<table border="1" class="dataframe">\n"""
    html_code += """  <thead>\n    <tr style="text-align: left;">\n"""
    for i in items[0]:
        html_code += f"      <th>{i}</th>\n"
    html_code += "    </tr>\n  </thead>\n  <tbody>\n"
    for line in items[1:]:
        html_code += "    <tr>\n"
        for elt in line:
            elt = f"{elt:.6f}" if isinstance(elt, float) else str(elt)
            html_code += f"      <td>{elt}</td>\n"
        html_code += "    </tr>\n"
    html_code += "  </tbody>\n</table><p>"
    return html_code


class NotebookProgressBar:
    """
    A progress par for display in a notebook.
    Class attributes (overridden by derived classes)
        - **warmup** (:obj:`int`) -- The number of iterations to do at the beginning while ignoring
          :obj:`update_every`.
        - **update_every** (:obj:`float`) -- Since calling the time takes some time, we only do it every presumed
          :obj:`update_every` seconds. The progress bar uses the average time passed up until now to guess the next
          value for which it will call the update.
    Args:
        total (:obj:`int`):
            The total number of iterations to reach.
        prefix (:obj:`str`, `optional`):
            A prefix to add before the progress bar.
        leave (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to leave the progress bar once it's completed. You can always call the
            :meth:`~transformers.utils.notebook.NotebookProgressBar.close` method to make the bar disappear.
        parent (:class:`~transformers.notebook.NotebookTrainingTracker`, `optional`):
            A parent object (like :class:`~transformers.utils.notebook.NotebookTrainingTracker`) that spawns progress
            bars and handle their display. If set, the object passed must have a :obj:`display()` method.
        width (:obj:`int`, `optional`, defaults to 300):
            The width (in pixels) that the bar will take.
    Example::
        import time
        pbar = NotebookProgressBar(100)
        for val in range(100):
            pbar.update(val)
            time.sleep(0.07)
        pbar.update(100)
    """

    warmup = 5
    update_every = 0.2

    def __init__(
        self,
        total: int,
        prefix: Optional[str] = None,
        leave: bool = True,
        parent: Optional["NotebookTrainingTracker"] = None,
        width: int = 300,
    ):
        self.total = total
        self.prefix = "" if prefix is None else prefix
        self.leave = leave
        self.parent = parent
        self.width = width
        self.last_value = None
        self.comment = None
        self.output = None

    def update(self, value: int, force_update: bool = False, comment: str = None):
        """
        The main method to update the progress bar to :obj:`value`.
        Args:
            value (:obj:`int`):
                The value to use. Must be between 0 and :obj:`total`.
            force_update (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force and update of the internal state and display (by default, the bar will wait for
                :obj:`value` to reach the value it predicted corresponds to a time of more than the :obj:`update_every`
                attribute since the last update to avoid adding boilerplate).
            comment (:obj:`str`, `optional`):
                A comment to add on the left of the progress bar.
        """
        self.value = value
        if comment is not None:
            self.comment = comment
        if self.last_value is None:
            self.start_time = self.last_time = time.time()
            self.start_value = self.last_value = value
            self.elapsed_time = self.predicted_remaining = None
            self.first_calls = self.warmup
            self.wait_for = 1
            self.update_bar(value)
        elif value <= self.last_value and not force_update:
            return
        elif (
            force_update
            or self.first_calls > 0
            or value >= min(self.last_value + self.wait_for, self.total)
        ):
            if self.first_calls > 0:
                self.first_calls -= 1
            current_time = time.time()
            self.elapsed_time = current_time - self.start_time
            self.average_time_per_item = self.elapsed_time / (value - self.start_value)
            if value >= self.total:
                value = self.total
                self.predicted_remaining = None
                if not self.leave:
                    self.close()
            else:
                self.predicted_remaining = self.average_time_per_item * (
                    self.total - value
                )
            self.update_bar(value)
            self.last_value = value
            self.last_time = current_time
            self.wait_for = max(int(self.update_every / self.average_time_per_item), 1)

    def update_bar(self, value, comment=None):
        spaced_value = " " * (len(str(self.total)) - len(str(value))) + str(value)
        if self.elapsed_time is None:
            self.label = f"[{spaced_value}/{self.total} : < :"
        elif self.predicted_remaining is None:
            self.label = (
                f"[{spaced_value}/{self.total} {format_time(self.elapsed_time)}"
            )
        else:
            self.label = f"[{spaced_value}/{self.total} {format_time(self.elapsed_time)} < {format_time(self.predicted_remaining)}"
            self.label += f", {1/self.average_time_per_item:.2f} it/s"
        self.label += (
            "]"
            if self.comment is None or len(self.comment) == 0
            else f", {self.comment}]"
        )
        self.display()

    def display(self):
        self.html_code = html_progress_bar(
            self.value, self.total, self.prefix, self.label, self.width
        )
        if self.parent is not None:
            # If this is a child bar, the parent will take care of the display.
            self.parent.display()
            return
        if self.output is None:
            self.output = disp.display(disp.HTML(self.html_code), display_id=True)
        else:
            self.output.update(disp.HTML(self.html_code))

    def close(self):
        "Closes the progress bar."
        if self.parent is None and self.output is not None:
            self.output.update(disp.HTML(""))


class NotebookTrainingTracker(NotebookProgressBar):
    """
    An object tracking the updates of an ongoing training with progress bars and a nice table reporting metrics.
    Args:
        num_steps (:obj:`int`): The number of steps during training.
        column_names (:obj:`List[str]`, `optional`):
            The list of column names for the metrics table (will be inferred from the first call to
            :meth:`~transformers.utils.notebook.NotebookTrainingTracker.write_line` if not set).
    """

    def __init__(self, num_steps, column_names=None, prefix=""):
        super().__init__(num_steps)
        self.inner_table = None if column_names is None else [column_names]
        self.child_bar = None
        self.prefix = prefix

    def display(self):
        self.html_code = html_progress_bar(
            self.value, self.total, self.prefix, self.label, self.width
        )
        if self.inner_table is not None:
            self.html_code += text_to_html_table(self.inner_table)
        if self.child_bar is not None:
            self.html_code += self.child_bar.html_code
        if self.output is None:
            self.output = disp.display(disp.HTML(self.html_code), display_id=True)
        else:
            self.output.update(disp.HTML(self.html_code))

    def write_line(self, values):
        """
        Write the values in the inner table.
        Args:
            values (:obj:`Dict[str, float]`): The values to display.
        """
        if self.inner_table is None:
            self.inner_table = [list(values.keys()), list(values.values())]
        else:
            columns = self.inner_table[0]
            if len(self.inner_table) == 1:
                # We give a chance to update the column names at the first iteration
                for key in values.keys():
                    if key not in columns:
                        columns.append(key)
                self.inner_table[0] = columns
            self.inner_table.append([values[c] for c in columns])

    def add_child(self, total, prefix=None, width=300):
        """
        Add a child progress bar displayed under the table of metrics. The child progress bar is returned (so it can be
        easily updated).
        Args:
            total (:obj:`int`): The number of iterations for the child progress bar.
            prefix (:obj:`str`, `optional`): A prefix to write on the left of the progress bar.
            width (:obj:`int`, `optional`, defaults to 300): The width (in pixels) of the progress bar.
        """
        self.child_bar = NotebookProgressBar(
            total, prefix=prefix, parent=self, width=width
        )
        return self.child_bar

    def remove_child(self):
        """
        Closes the child progress bar.
        """
        self.child_bar = None
        self.display()


##################################################################################################################
# End of copy
###################################################################################################################


class NotebookProgressCallback(ProgressBarBase):
    def __init__(self):
        super().__init__()
        self._force_next_update = False
        self.training_tracker = None
        self.prediction_bar = None

    def on_sanity_check_start(self, trainer, pl_module):
        super().on_sanity_check_start(trainer, pl_module)
        # dummy progress bar
        self.prediction_bar = NotebookProgressBar(int(trainer.num_sanity_val_steps), prefix="Validation sanity check")

    def on_sanity_check_end(self, trainer, pl_module):
        super().on_sanity_check_end(trainer, pl_module)
        # remove bars
        self.prediction_bar.update(1)
        self.prediction_bar.close()
        self.training_tracker = None
        self.prediction_bar = None

    def on_train_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        self.first_column = "Epoch"
        steps = trainer.max_steps or int(self.total_train_batches * trainer.max_epochs)
        self.training_tracker = NotebookTrainingTracker(steps, prefix="Training")

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        self.start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        prog_dict = self._format_prog_bar_dict(trainer.progress_bar_dict)
        comment = f"Epoch {int(trainer.current_epoch)} {prog_dict}"
        step = trainer.global_step + 1
        self.training_tracker.update(step, comment=comment, force_update=self._force_next_update,)
        self._force_next_update = False

    def on_validation_start(self, trainer, pl_module):
        super().on_validation_start(trainer, pl_module)
        if not trainer.running_sanity_check:
            if self.prediction_bar is None:
                if self.training_tracker is not None:
                    self.prediction_bar = self.training_tracker.add_child(self.total_val_batches, prefix="Validating")
        self.prediction_bar.update(1)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self.prediction_bar.update(self.prediction_bar.value + 1)

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        total_train_batches = self.total_train_batches
        metrics = trainer.callback_metrics
        values = {}
        values["epoch"] = trainer.current_epoch

        if self.training_tracker is not None:
            for k, v in metrics.items():
                name = str(k)
                if isinstance(v, torch.Tensor):
                    values[name] = v.data.cpu().numpy().item()
                else:
                    values[name] = v

            if total_train_batches != float("inf"):
                # Measure speed performance metrics.
                runtime = time.time() - self.start_time  # seconds
                total_val_batches = self.total_val_batches
                n_obs = total_train_batches + total_val_batches
                samples_per_second = 1 / (runtime / n_obs)

                values["time"] = round(runtime, 4)
                values["samples/s"] = round(samples_per_second, 4)

            self.training_tracker.write_line(values)
            self.training_tracker.remove_child()
            self.prediction_bar = None
            self._force_next_update = True

    def on_train_epoch_end(self, trainer, pl_module, outputs: Any) -> None:
        super().on_train_epoch_end(trainer, pl_module, outputs)
        if trainer.val_dataloaders is None:
            self.on_validation_end(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        self.training_tracker.update(trainer.global_step, force_update=True)
        self.training_tracker = None

    def on_test_start(self, trainer, pl_module):
        super().on_test_start(trainer, pl_module)
        self.test_progress_bar = NotebookProgressBar(int(self.total_test_batches), prefix="Testing")
        self.test_progress_bar.update(1)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self.test_progress_bar.update(self.test_progress_bar.value + 1)

    def on_test_end(self, trainer, pl_module):
        super().on_test_end(trainer, pl_module)
        self.test_progress_bar.close()

    def on_predict_start(self, trainer, pl_module):
        super().on_predict_start(trainer, pl_module)
        self.predict_progress_bar = NotebookProgressBar(int(self.total_test_batches), prefix="Testing")
        self.predict_progress_bar.update(1)
        
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_predict_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self.predict_progress_bar.update(self.test_progress_bar.value + 1)
        
    def on_predict_end(self, trainer, pl_module):
        self.predict_progress_bar.close()        

    @staticmethod
    def _format_prog_bar_dict(progbar_dict: dict):
        vals = {}
        for k, v in progbar_dict.items():
            if isinstance(v, torch.Tensor):
                v = round(v.data.cpu().numpy().item(), 4)
            elif isinstance(v, str):
                pass
            else:
                v = round(v, 3)
            vals[k] = v
        return vals
