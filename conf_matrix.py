from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def conf_matrix(y-pred, y-test):
  gt_label = np.unique(labels) #need all the POS tags
  conf_mat = confusion_matrix(y_test, y_pred, labels=gt_label, normalize='true') #predicted labels and actual labels
  class_correct = np.diag(conf_mat)

  fig, ax = plt.subplots(figsize=(10, 10))
  disp = ConfusionMatrixDisplay(conf_mat, display_labels=gt_label)
  disp.plot(include_values=False, xticks_rotation='vertical', ax=ax, cmap=plt.cm.plasma)
  disp.ax_.get_images()[0].set_clim(0, 1)
  ax.set_title('Confusion matrix')
  fig.tight_layout()
  plt.savefig('Conf_mat.png')
