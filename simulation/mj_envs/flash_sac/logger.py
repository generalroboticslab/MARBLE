"""Compact console logger for FlashSAC + the shared template-rendering engine.

The flash_sac package carries no rsl_rl dependency (portability):
  - _TemplateLogMixin: section/format template engine, built once per unique extras key set.
  - FlashSACCompactLogger: standalone logger (all data passed as args; no rsl_rl base class).
"""

import time

import tqdm as tqdm_module


class _TemplateLogMixin:
    """Shared template-building engine for CompactLogger and FlashSACCompactLogger.

    Parses a flat dict of prefixed keys (section/name or section/subsection/name)
    into a list of rendering instructions built once per unique key set.  Subsequent
    iterations substitute values into pre-built format strings with no sorting or
    grouping overhead.

    Template entries:
      ('s', static_text)       — append fixed string
      ('v', fmt_str, [keys])   — append fmt_str.format(*[extras[k] for k in keys])
    """

    def _rebuild_template(
        self,
        extras_dict: dict[str, float],
        width: int,
        print_minimal: bool = False,
    ) -> None:
        """Build rendering template from the current extras key set.

        Called once per unique key set (typically once per run).
        print_minimal: when True, skip all sections except Episode_Termination.
        """
        self._template_keys: frozenset = frozenset(extras_dict)
        self._n_keys: int = len(extras_dict)
        scalar_col_w = 32
        range_col_w = 26

        # Build section hierarchy: sections[section][subsection][name] = full_key
        sections: dict[str, dict[str | None, dict[str, str]]] = {}
        for key in sorted(extras_dict):
            parts = key.split("/")
            if len(parts) == 2:
                section, subsection, name = parts[0], None, parts[1]
            elif len(parts) >= 3:
                section, subsection, name = parts[0], parts[1], "/".join(parts[2:])
            else:
                continue  # bare keys (no "/") not renderable — skip
            sections.setdefault(section, {}).setdefault(subsection, {})[name] = key

        template: list[tuple] = []
        for section in sorted(sections):
            if print_minimal and section not in ["Episode_Termination"]:
                continue
            subsections = sections[section]
            template.append(("s", f"{section}:\n"))
            if None in subsections:
                self._append_h2_section(
                    template, subsections[None], section, None, width,
                    scalar_col_w=scalar_col_w, range_col_w=range_col_w,
                )
            for subsection in sorted(k for k in subsections if k is not None):
                label = "DR:" if subsection == "domain_randomization" else f"{subsection}:"
                template.append(("s", label + "\n"))
                self._append_h2_section(
                    template, subsections[subsection], section, subsection, width,
                    scalar_col_w=scalar_col_w, range_col_w=range_col_w,
                    strip_range_suffix=(subsection == "domain_randomization"),
                )
        self._template = template

    def _append_h2_section(
        self,
        template: list[tuple],
        items: dict[str, str],
        section: str,
        subsection: str | None,
        width: int,
        indent: str = "",
        scalar_col_w: int = 32,
        range_col_w: int = 26,
        strip_range_suffix: bool = False,
    ) -> None:
        """Append one section/subsection group to the template.

        Detects _min/_max suffix pairs and renders them as range rows [min, max].
        All other entries render as scalar rows.  Both layouts use 2-per-line packing.
        strip_range_suffix: strip trailing _range/_ranges from display labels (DR sections).
        """
        if not items:
            return

        scalars: list[tuple[str, str]] = []          # (display_label, tb_key)
        ranges: list[tuple[str, str, str]] = []      # (display_label, min_key, max_key)
        rendered: set[str] = set()

        for name in sorted(items):
            if name in rendered:
                continue
            min_name = max_name = None
            if name.endswith("_min"):
                base = name[:-4]
                cand = f"{base}_max"
                if cand in items:
                    min_name, max_name = name, cand
            elif name.endswith("_max"):
                base = name[:-4]
                cand = f"{base}_min"
                if cand in items:
                    min_name, max_name = cand, name
            if min_name and max_name:
                ranges.append((base, items[min_name], items[max_name]))
                rendered |= {min_name, max_name}
            else:
                scalars.append((name, items[name]))
                rendered.add(name)

        def _strip(label: str) -> str:
            if strip_range_suffix:
                if label.endswith("_ranges"):
                    return label[:-7]
                if label.endswith("_range"):
                    return label[:-6]
            return label

        for i in range(0, len(scalars), 2):
            label1, key1 = scalars[i]
            label1 = _strip(label1)
            if i + 1 < len(scalars):
                label2, key2 = scalars[i + 1]
                label2 = _strip(label2)
                fmt = f"{indent}{label1:<{scalar_col_w}} {{:>9.4f}}    {label2:<{scalar_col_w}} {{:>9.4f}}\n"
                template.append(("v", fmt, [key1, key2]))
            else:
                fmt = f"{indent}{label1:<{scalar_col_w}} {{:>9.4f}}\n"
                template.append(("v", fmt, [key1]))

        for i in range(0, len(ranges), 2):
            label1, min_key1, max_key1 = ranges[i]
            label1 = _strip(label1)
            if i + 1 < len(ranges):
                label2, min_key2, max_key2 = ranges[i + 1]
                label2 = _strip(label2)
                fmt = f"{indent}{label1:<{range_col_w}} [{{:>7.3g}},{{:>7.3g}}]  {label2:<{range_col_w}} [{{:>7.3g}},{{:>7.3g}}]\n"
                template.append(("v", fmt, [min_key1, max_key1, min_key2, max_key2]))
            else:
                fmt = f"{indent}{label1:<{range_col_w}} [{{:>7.3g}},{{:>7.3g}}]\n"
                template.append(("v", fmt, [min_key1, max_key1]))


class FlashSACCompactLogger(_TemplateLogMixin):
    """Compact console logger for FlashSAC training (standalone, no rsl_rl dependency).

    Uses the same template-rendering engine as CompactLogger but is decoupled from
    rsl_rl's Logger base class. All training data is passed as arguments to log_flash_sac();
    no internal state accumulation beyond the cached rendering template.

    Template is rebuilt only when the extras key set changes (O(1) fast path otherwise).
    """

    def __init__(self, width: int = 90) -> None:
        self._width = width
        self._template: list[tuple] = []
        self._template_keys: frozenset = frozenset()
        self._n_keys: int = -1

    def log_flash_sac(
        self,
        it: int,
        total_it: int,
        collect_time: float,
        learn_time: float,
        metric_avgs: dict,
        mean_rew: float | None,
        mean_len: float | None,
        extras_log: dict[str, float],
        total_steps: int,
        fps: float,
        tot_time: float,
        iter_time: float,
        start_it: int,
    ) -> None:
        """Print compact console output for one FlashSAC logging interval.

        Args:
            metric_avgs: averaged training metrics — qf_loss, actor_loss, alpha_value,
                         alpha_loss, policy_entropy, action_std, qf_max, qf_min, etc.
            extras_log: flat dict of all section scalars with prefixed keys:
                Episode_Reward/*, Curriculum/*, DR/*_min/max, Metrics/*, Episode_Termination/*.
                Template engine sections from key prefixes automatically.
            iter_time: wall time for one logging interval (collect + learn).
        """
        width = self._width
        extras_dict = extras_log

        if (len(extras_dict) != self._n_keys
                or frozenset(extras_dict) != self._template_keys):
            self._rebuild_template(extras_dict, width)

        critic_loss = metric_avgs.get("qf_loss", 0.0)
        actor_loss = metric_avgs.get("actor_loss", 0.0)
        entropy = metric_avgs.get("policy_entropy", 0.0)
        alpha = metric_avgs.get("alpha_value", 0.0)
        noise = metric_avgs.get("action_std", 0.0)
        mean_rew_str = f"{mean_rew:7.2f}" if mean_rew is not None else "    N/A"
        mean_len_str = f"{mean_len:6.2f}" if mean_len is not None else "   N/A"

        iter_label = f" iter {it}/{total_it} "
        left = (width - len(iter_label)) // 2
        right = width - len(iter_label) - left

        metrics_line = (
            f" collect {collect_time:6.3f}s  learn {learn_time:6.3f}s"
            f"  │  critic {critic_loss:7.4f}  actor {actor_loss:7.4f}"
            f"  entropy {entropy:7.4f}  alpha {alpha:.6f}"
        )
        if "estimator_loss" in metric_avgs:
            metrics_line += f"  estimator {metric_avgs['estimator_loss']:7.4f}"
        if "temporal_max_weight" in metric_avgs:
            metrics_line += f"  temporal_max_weight {metric_avgs['temporal_max_weight']:.4f}"
        metrics_line += "\n"

        parts = [
            f"{'═' * left}{iter_label}{'═' * right}\n",
            metrics_line,
            f" reward {mean_rew_str}  eplen {mean_len_str}"
            f"  │  noise {noise:5.2f}  │  {total_steps} steps  {fps:.0f} fps\n",
        ]

        for kind, *args in self._template:
            if kind == "s":
                parts.append(args[0])
            else:
                fmt, keys = args[0], args[1]
                parts.append(fmt.format(*[extras_dict.get(k, 0.0) for k in keys]))

        done_it = max(it - start_it, 1)
        remaining_it = total_it - it
        eta = tot_time / done_it * remaining_it if done_it > 0 else 0.0
        parts.extend([
            f" iter {iter_time:5.2f}s  │  elapsed {time.strftime('%H:%M:%S', time.gmtime(tot_time))}"
            f"  │  ETA {time.strftime('%H:%M:%S', time.gmtime(eta))}\n",
            f"{'═' * width}\n",
        ])

        tqdm_module.tqdm.write("".join(parts))
