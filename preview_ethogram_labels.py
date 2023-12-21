import h5py

import matplotlib.pyplot as plt
import utils


prediction_file_path = "/media/erato_0/Data1/ethogram_workspace/walk_touch_floor_deepethogram/DATA/05072023_rat_s1_n_1_000/05072023_rat_s1_n_1_000_outputs.h5"
bout_percentiles_file_path = "/media/erato_0/Data1/ethogram_workspace/walk_touch_floor_deepethogram/DATA/bout_percentiles.h5"

with h5py.File(bout_percentiles_file_path, "r") as f:
    percentiles = f['percentiles'][()]

probabilities, thresholds = utils.read_ethogram_result(prediction_file_path, model='resnet18')
predictions = utils.get_ethogram_predictions(
    percentiles,
    probabilities,
    thresholds
)

step_predictions = (probabilities[:, 3] > 0.7).astype(int)

fig = plt.figure(figsize=(20, 4))
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)

ax1.plot(probabilities[:, 0])
ax1.plot(predictions[:, 0])
ax1.set_ylabel("background")

ax2.plot(probabilities[:, 1])
ax2.plot(predictions[:, 1])
ax2.set_ylabel("walk")

ax3.plot(probabilities[:, 2])
ax3.plot(predictions[:, 2])
ax3.set_ylabel("floor")

ax4.plot(probabilities[:, 3])
ax4.plot(step_predictions)
ax4.set_ylabel("steps")

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_ylim(-0.2, 1.2)

fig.tight_layout()
plt.show()