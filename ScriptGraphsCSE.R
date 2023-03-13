#Code for graphs CSE - Midterm

models <- c("Logistic regression", "MLP", "HMM")
accuracies <- c(92, 91, 91)

# combine the vectors into a dataframe
data <- data.frame(Model = models, Accuracy = accuracies)

#barplot
ggplot(data, aes(x = Model, y = Accuracy, fill = Model)) + 
  geom_col(stat = "dodge", position = position_dodge(.9)) + 
  geom_text(aes(label = paste0(Accuracy, " %")), 
            position = position_dodge(width = .9), 
            vjust = -0.5, size = 3.5) +
  scale_y_continuous(expand = c(0, 0), limits = c(0, 100)) +
  xlab("Model") + ylab("Accuracy (%)") + 
  labs(title = "Final Model Accuracies") + 
  theme(plot.title = element_text(hjust = 0.5),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 14),
        axis.title.x = element_text(margin = margin(t = 10))) +
  scale_fill_manual(values = c("#FFA07A", "#87CEFA", "#9ACD32"))



df <- data.frame(models <- c( "HMM", "+ lowercase", "+ glove-50", "+ glove-100", "+ fasttext", "- lowercase", "+ topn=1000"),
                 Accuracy = c(91.43, 90.47, 91.55, 91.71, 91.65, 92.51, 92.46))


ggplot(df, aes(x = Model, y = Accuracy)) + 
  geom_point(size = 4, color = "#FFA07A") +
  geom_text(aes(label = paste0(Accuracy, " %")), position = position_nudge(y = 3), size = 3.5) +
  geom_line(linewidth = 0.5, color = "blue") +
  geom_hline(aes(yintercept = 92.14, color = "Baseline"), show.legend = TRUE, linetype = "dashed") + 
  geom_vline(xintercept = seq_along(df$Model), linetype = "dotted", color = "lightgrey") + 
  scale_y_continuous(expand = c(0, 0), limits = c(0, 100)) +
  xlab("Model Hyperparameter") + ylab("Accuracy (%)") + 
  labs(title = "Model Accuracies") +
  scale_color_manual(name = "Legend", values = c("Baseline" = "lightblue")) +
  theme(plot.title = element_text(hjust = 0.5),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 14),
        axis.title.x = element_text(margin = margin(t = 10)))
