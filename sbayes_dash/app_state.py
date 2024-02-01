from __future__ import annotations

import io
from base64 import b64encode

from dataclasses import dataclass

import numpy as np
from matplotlib import colors as mpl_colors

from sbayes_dash.util import COLOR_0


@dataclass
class AppState:

    clusters_path = None
    _clusters = None
    data_fig = None
    fig = None
    lines = None
    scatter = None
    cluster_colors = None
    locations = None
    object_data = None
    objects = None
    i_sample = 0
    burnin = 0
    data = None
    data_crs: str | None = None

    highlighted_cluster = None
    summary = True

    @property
    def clusters(self):
        return self._clusters

    @clusters.setter
    def clusters(self, clusters):
        self._clusters = clusters
        self.cluster_colors = np.array(self.get_cluster_colors(self.n_clusters))

    @staticmethod
    def get_cluster_colors(k: int) -> list[str]:
        # cm = plt.get_cmap('gist_rainbow')
        # cluster_colors = [colors.to_hex(c) for c in cm(np.linspace(0, 1, k, endpoint=False))]
        colors = []
        for i, x in enumerate(np.linspace(0, 1, k, endpoint=False)):
            b = i % 2
            h = x % 1
            s = 0.6 + 0.4 * b
            v = 0.5 + 0.3 * (1 - b)
            r, g, b = mpl_colors.hsv_to_rgb((h, s, v))
            colors.append(mpl_colors.to_hex((r,g,b)))
            # colors.append(f"rgba({r*255:.0f}, {g*255:.0f}, {b*255:.0f}, 255)")
        colors.append(COLOR_0)
        return colors

    @property
    def n_clusters(self) -> int:
        return self.clusters.shape[0]

    @property
    def n_samples(self) -> int:
        return self.clusters.shape[1]

    @property
    def n_objects(self):
        return self.objects.n_objects

    def serialize_results_map(self, filename: str) -> dict:
        buffer = io.StringIO()
        self.fig.write_html(buffer)
        html_bytes = buffer.getvalue().encode()
        content = b64encode(html_bytes).decode()
        return {
            "base64": True,
            "content": content,
            "type": "text/html",
            "filename": filename
        }
