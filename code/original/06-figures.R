##################################
## Replication Code for Figures ##
##################################

library(ggplot2)
library(ggthemes)
library(dplyr)
library(tidyr)
library(RColorBrewer)
library(stargazer)

## Set working directory to the location of the master "word2vec_time" folder

setwd("")

######################
## Colors and themes

# Set colors/pastels and store in vector
blue <- "#67a9cf"
green <- "#008837"
purple <- "#5e3c99"
peach <- "#fb6a4a"
crimson <- "#cb181d"
colorVec <- c(blue, green, purple, peach, crimson)

# Create theme for figures (Chris Adolph's Golden Scatterplot theme)
goldenScatterCAtheme <- theme(
  ## Removes main plot gray background
  panel.background = element_rect(fill = "white"), 
  
  ## Golden rectangle plotting area (leave out for square)
  aspect.ratio = ((1 + sqrt(5))/2)^(-1), 
  
  ## All axes changes
  axis.ticks.length = unit(0.5, "char"),  # longer ticks
  
  ## Horizontal axis changes
  axis.line.x.top = element_line(size = 0.2),    # thinner axis lines
  axis.line.x.bottom = element_line(size = 0.2), # thinner axis lines
  axis.ticks.x = element_line(size = 0.2),       # thinner ticks
  axis.text.x = element_text(color = "black", size = 12),
  ## match type of axis labels and titles
  axis.title.x = element_text(size = 12,
                              margin = margin(t = 7.5, r = 0, b = 0, l = 0)),
  ## match type; pad space between title and labels
  
  ## Vertical axis changes
  axis.ticks.y = element_blank(), # no y axis ticks (gridlines suffice)
  axis.text.y = element_text(color = "black", size = 12,
                             margin = margin(t = 0, r = -4, b = 0, l = 0)),
  ## match type of axis labels and titles, pad
  axis.title.y = element_text(size = 12,
                              margin = margin(t = 0, r = 7.5, b = 0, l = 0)),
  ## match type of axis labels and titles, pad
  
  ## Legend
  legend.key = element_rect(fill = NA, color = NA),
  ## Remove unhelpful gray background
  
  ## Gridlines (in this case, horizontal from left axis only
  panel.grid.major.x = element_blank(),
  panel.grid.major.y = element_line(color = "gray45", size = 0.2),
  
  ## Faceting (small multiples)
  strip.background = element_blank(),
  ## Remove unhelpful trellis-like shading of titles
  strip.text.x = element_text(size=12),  # Larger facet titles
  strip.text.y = element_blank(),        # No titles for rows of facets
  strip.placement = "outside",           # Place titles outside plot
  panel.spacing.x = unit(1.25, "lines"), # Horizontal space b/w plots
  panel.spacing.y = unit(1, "lines")     # Vertical space b/w plots
)
class(goldenScatterCAtheme)  ## Creates this as a class


######################
## Import gold standard and word2vec model data

# Data from gold standard model of corpus
supervised <- read.csv("./data/supervised_category_means.csv") %>%   #eras 2-6
                      select(-X) %>%
                      rename(prop = mean)
hand_coded <- read.csv("./codebook/hand_coded.csv") %>%              #era 1
                      select(code) %>%
                      rename(category = code)
era1_count <- as.numeric(length(hand_coded[,1]))
hand_coded <- hand_coded %>% group_by(category) %>% 
                             tally() %>%
                             mutate(prop = n/era1_count) %>%
                             select(category, prop) %>%
                             mutate(era = 1) %>% 
                             filter(category == 20 | 
                                    category == 40 |
                                    category == 60 |
                                    category == 61)
german <- c(41, 0, 1)
hand_coded <- rbind(hand_coded, german) %>% arrange(category) %>% select (era, category, prop)

gold_standard <- rbind(hand_coded, supervised) %>% arrange(era, category)

# Data from naive time word2vec model of corpus
naive <- read.csv("./data/naive_mean_output.csv", check.names = FALSE, header = TRUE)
naive <- naive %>% mutate(category = c(20, 40, 41, 60, 61))
naive <- naive %>% gather(key = "year", value = "1855:2005", -category) %>% 
                   rename("mean" = "1855:2005")

# Data from overlapping word2vec model of corpus
overlap <- read.csv("./data/overlap_mean_output.csv", check.names = FALSE, header = TRUE)
overlap <- overlap %>% mutate(category = c(20, 40, 41, 60, 61))
overlap <- overlap %>% gather(key = "year", value = "1855:2005", -category) %>% 
  rename("mean" = "1855:2005")

# Data from aligned word2vec model of corpus
aligned <- read.csv("./data/aligned_mean_output.csv", check.names = FALSE, header = TRUE)
aligned <- aligned %>% mutate(category = c(20, 40, 41, 60, 61))
aligned <- aligned %>% gather(key = "year", value = "1880:2005", -category) %>% 
  rename("mean" = "1880:2005")
aligned <- rbind(filter(naive, year == "1855"), aligned)

# Data from chronologically trained word2vec model of corpus
chrono <- read.csv("./data/chrono_mean_output.csv", check.names = FALSE, header = TRUE)
chrono <- chrono %>% mutate(category = c("20", "40", "41", "60", "61", "social"))
chrono <- chrono %>% gather(key = "year", value = "1855:2005", -category) %>% 
  rename("mean" = "1855:2005") %>% filter(category != "social") %>% mutate(year = as.numeric(year))

# Data from "Social" - "Equality" Output (from chronological model)

social <- read.csv("./data/chrono_social_output.csv", header = FALSE) %>%
              rename(mean = V1, lower = V2, upper = V3) %>%
              mutate(era = seq(1, 7, 1))

######################
## Figure 1 

# Data frame for annotations of lines
cat_labels  <- data.frame(era=rep(7.1, 5),
                       baseline=gold_standard$prop[gold_standard$era==7],
                       category=c("Gender", "International Relations", "Germany",
                                  "Race", "African American"),
                       labels=c("Gender", "Int'l. Relations", "Germany",
                                "Race", "African\nAmerican"))

cat_labels[4,2] <- cat_labels[4,2] + .02    #adjusting race label

# Initialize ggplot
gold_standard_plot <- ggplot(gold_standard, aes(x=era, y=prop)) +
  goldenScatterCAtheme +
  geom_line(alpha = .6, 
            aes(colour = factor(category, 
                                labels = c("Gender", "Int'l. Relations", "Germany",
                                           "Race", "African Amer."))), size = .75) +
  labs(color = "Topics", x = "", y = "Proportion of Articles") +
  scale_colour_manual(values=colorVec, guide="none") +  
  scale_x_continuous(breaks = seq(from = 1, to = 7, by = 1),
                     labels = c("1855-\n1879", "1880-\n1904", "1905-\n1929", "1930-\n1954", 
                                "1955-\n1979", "1980-\n2004", "2005-\n2016"),
                     limits = c(1, 8)) +
  geom_text(data=cat_labels, mapping=aes(x=era, y=baseline, label=labels), 
            alpha=0.9, hjust = 0, size=3.5, lineheight = .75) +
  theme(legend.position="none", axis.text.x = element_text(color = "black", size = 10))
gold_standard_plot

ggsave(filename = "fig1.pdf", plot = gold_standard_plot, width = 7, path = "./figures/")

######################
## Figure 2 

## Naive model sub-figure

naive_plot <- ggplot(naive, aes(x=as.numeric(year), y=mean, color=as.character(category))) +
  goldenScatterCAtheme +
  geom_line(alpha = .7, size = 1, position=position_dodge(width=1)) + 
  labs(color = "Terms", x = "", y = "Cosine Similarity w/ Equality") +
  scale_x_continuous(breaks = seq(1855, 2005, 25),
                     labels = c("1855-\n1879", "1880-\n1904", "1905-\n1929", "1930-\n1954", 
                                "1955-\n1979", "1980-\n2004", "2005-\n2016"),
                     limits = c(1854, 2006)) +
  scale_y_continuous(limits = c(0, .71),
                     breaks = seq(.1, .7, .2)) +
  scale_color_manual(values=colorVec, guide="none") +
  ggtitle("Naive Time Model") +
  theme(plot.title = element_text(hjust = 0.5), 
        axis.text.x = element_text(color = "black", size = 10))
naive_plot

ggsave(filename = "fig2_naive.pdf", plot = naive_plot, 
       width = 5, height = 4, path = "./figures/")

## Overlapping model sub-figure

overlap_plot <- ggplot(overlap, aes(x=as.numeric(year), y=mean, color=as.character(category))) +
  goldenScatterCAtheme +
  geom_line(alpha = .7, size = 1, position=position_dodge(width=1)) + 
  labs(color = "Words", x = "", y = "") +
  scale_x_continuous(breaks = seq(1855, 2005, 25),
                     labels = c("1855-\n1879", "1880-\n1904", "1905-\n1929", "1930-\n1954", 
                                "1955-\n1979", "1980-\n2004", "2005-\n2016"),
                     limits = c(1854, 2006)) +
  scale_y_continuous(limits = c(0, .71),
                     breaks = seq(.1, .7, .2)) +
  scale_color_manual(values=colorVec, guide="none") +
  ggtitle("Overlapping Model") +
  theme(plot.title = element_text(hjust = 0.5), 
        axis.text.x = element_text(color = "black", size = 10))
overlap_plot

ggsave(filename = "fig2_overlap.pdf", plot = overlap_plot, 
       width = 5, height = 4, path = "./figures/")

## Aligned model sub-figure

aligned_plot <- ggplot(aligned, aes(x=as.numeric(year), y=mean, color=as.character(category))) +
  goldenScatterCAtheme +
  geom_line(alpha = .7, size = 1, position=position_dodge(width=1)) + 
  labs(color = "Terms", x = "", y = "Cosine Similarity w/ Equality") +
  scale_x_continuous(breaks = seq(1855, 2005, 25),
                     labels = c("1855-\n1879", "1880-\n1904", "1905-\n1929", "1930-\n1954", 
                                "1955-\n1979", "1980-\n2004", "2005-\n2016"),
                     limits = c(1854, 2006)) +
  scale_y_continuous(limits = c(0, .71),
                     breaks = seq(.1, .7, .2)) +
  scale_color_manual(values=colorVec, guide="none") +
  ggtitle("Aligned Model") +
  theme(plot.title = element_text(hjust = 0.5), 
        axis.text.x = element_text(color = "black", size = 10))
aligned_plot

ggsave(filename = "fig2_aligned.pdf", plot = aligned_plot, 
       width = 5, height = 4, path = "./figures/")

## Chronological model sub-figure
# Create data frame to annotate lines from chronological model
chrono_labels  <- data.frame(era=rep(2007, 5),
                          baseline=chrono$mean[chrono$year==2005],
                          category=c("20", "40", "41", "60", "61"),
                          labels=c("Gender", "Int'l. Relations", "Germany",
                                   "Race", "African\nAmerican"))

chrono_labels[3,2] <- chrono_labels[3,2] + .014    #adjusting Germany label
chrono_labels[5,2] <- chrono_labels[5,2] - .04    #adjusting African Amer. label

# Initializing ggplot
chrono_plot <- ggplot(chrono, aes(x=year, y=mean, color=category)) +
  goldenScatterCAtheme +
  geom_line(alpha = .7, size = 1, position=position_dodge(width=1)) + 
  labs(color = "Terms", x = "", y = "") +
  scale_x_continuous(breaks = seq(1855, 2005, 25),
                     labels = c("1855-\n1879", "1880-\n1904", "1905-\n1929", "1930-\n1954", 
                                "1955-\n1979", "1980-\n2004", "2005-\n2016"),
                     limits = c(1854, 2040)) +
  scale_y_continuous(limits = c(0, .71),
                     breaks = seq(.1, .7, .2)) +
  scale_color_manual(values=colorVec, guide="none") +
  ggtitle("Chronologically Trained Model") +
  geom_text(data=chrono_labels, mapping=aes(x=era, y=baseline, label=labels), 
            alpha=0.9, hjust = 0, size=3.5, lineheight = .75) +
  theme(plot.title = element_text(hjust = 0.5), legend.position="none", 
        axis.text.x = element_text(color = "black", size = 10))
chrono_plot

ggsave(filename = "fig2_chrono.pdf", plot = chrono_plot, 
       width = 5, height = 4, path = "./figures/")


######################
## Figure 3

# Create data frame for figure 3
fig3_data <- gold_standard %>% rename("baseline" = "prop")
fig3_data <- cbind(fig3_data, select(chrono, mean)) 
fig3_data <- fig3_data %>% rename("model_value" = "mean") %>%
                           mutate(sq_model_value = (model_value*100)^2) 

z_baseline <- scale(fig3_data$baseline)
z_model_value <- scale(fig3_data$model_value)
z_sq_model_value <- scale(fig3_data$sq_model_value)

fig3_data <- cbind(fig3_data, z_baseline, z_model_value, z_sq_model_value)

# Rename the category variable for use as facet titles
fig3_data$category <- plyr::mapvalues(fig3_data$category,
                                      from = c("20", "40", "41", "60", "61"),
                                      to = c("Gender",
                                             "International Relations",
                                             "Germany",
                                             "Race",
                                             "African American"))

# Reorder the categories to change the order of plots
fig3_data$category <- factor(fig3_data$category,
                              c("Race",
                                "African American",
                                "International Relations",
                                "Germany",
                                "Gender"))

# Data frame for annotations of word2vec line
notesW2V <- data.frame(era=rep(7.5, 5),
                       model_value=fig3_data$z_model_value[fig3_data$era==7],
                       category=c("Gender", "International Relations", "Germany",
                                  "Race", "African American"),
                       labels=rep("W2V", 5))

# Data frame for annotations of gold standard
notesGS  <- data.frame(era=rep(7.5, 5),
                       baseline=fig3_data$z_baseline[fig3_data$era==7],
                       category=c("Gender", "International Relations", "Germany",
                                  "Race", "African American"),
                       labels=rep("GS", 5))

# Initialize ggplot
fig3 <- ggplot(fig3_data, aes(x = era, y = z_model_value, color=category)) +
  goldenScatterCAtheme +
  facet_wrap(~category, nrow=3, ncol=2) +
  geom_line(size = 1.25, alpha = 0.9) +
  geom_line(aes(y = z_baseline), size = 1.5, alpha = 0.3) +
  labs(x = "", y = "z-score Normalized Model Output") +
  scale_x_continuous(breaks = seq(1, 7, 1),
                     labels = c("1855-\n1879 ", "1880-\n1904 ", "1905-\n1929 ", "1930-\n1954 ", 
                                "1955-\n1979 ", "1980-\n2004 ", "2005-\n2016 "),
                     limits = c(1, 8.5)) +
  scale_y_continuous(limits = c(-1.65, 3.45)) +
  scale_color_manual(values=c("#fb6a4a", "#cb181d", "#008837","#5e3c99","#67a9cf"), guide="none") +
  geom_text(data=notesW2V, mapping=aes(x=era, y=model_value, label=labels), alpha=0.9, size=4) +
  geom_text(data=notesGS, mapping=aes(x=era, y=baseline, label=labels), alpha=0.6, size=4) +
  theme(legend.position="none", axis.text.x = element_text(color = "black", size = 10))
fig3 

ggsave(filename = "fig3.pdf", plot = fig3, width=8.5, height=10, path = "./figures/")

######################
## Figure 4

social_plot <- ggplot(social, aes(x=era, y=mean)) + 
  goldenScatterCAtheme +
  geom_line() +
  geom_point(size = .5) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = .25, alpha = .5) +
  labs(x = "", y = "Cosine Similarity") +
  scale_x_continuous(breaks = seq(1, 7, 1),
                     labels = c("1855-\n1879", "1880-\n1904", "1905-\n1929", "1930-\n1954", 
                                "1955-\n1979", "1980-\n2004", "2005-\n2016")) +
  ylim(0.15, .5)
social_plot

ggsave(filename = "fig4.pdf", plot = social_plot, width = 7, path = "./figures/")

######################
## Table 2 and ANOVA

# Data for table 2, statistical comparison of baseline to word2vec models
table2_data <- fig3_data %>% rename(z_chrono = z_model_value,
                                    z_chrono_sq = z_sq_model_value,
                                    chrono = model_value,
                                    chrono_sq = sq_model_value) %>%
                             mutate(naive = naive$mean,
                                    overlap = overlap$mean,
                                    aligned = aligned$mean) %>%
                             mutate(naive_sq = (naive*100)^2,
                                    overlap_sq = (overlap*100)^2,
                                    aligned_sq = (aligned*100)^2) 

z_naive <- scale(table2_data$naive)
z_overlap <- scale(table2_data$overlap)
z_aligned <- scale(table2_data$aligned)
z_naive_sq <- scale(table2_data$naive_sq)
z_overlap_sq <- scale(table2_data$overlap_sq)
z_aligned_sq <- scale(table2_data$aligned_sq)

table2_data <- cbind(table2_data, z_naive, z_overlap, z_aligned,
                     z_naive_sq, z_overlap_sq, z_aligned_sq)
  
table2_data <- table2_data %>% mutate(chrono_dev = abs(z_baseline - z_chrono),
                                    chrono_sq_dev = abs(z_baseline - z_chrono_sq),
                                    naive_dev = abs(z_baseline - z_naive),
                                    naive_sq_dev = abs(z_baseline - z_naive_sq),
                                    overlap_dev = abs(z_baseline - z_overlap),
                                    overlap_sq_dev = abs(z_baseline - z_overlap_sq),
                                    aligned_dev = abs(z_baseline - z_aligned),
                                    aligned_sq_dev = abs(z_baseline - z_aligned_sq))

# Calculating variance, squared variance, and correlation
dev_scores <- c(sum(table2_data$naive_dev), sum(table2_data$overlap_dev),
                sum(table2_data$chrono_dev), sum(table2_data$aligned_dev))

sq_dev_scores <- c(sum(table2_data$naive_sq_dev), sum(table2_data$overlap_sq_dev),
                   sum(table2_data$chrono_sq_dev), sum(table2_data$aligned_sq_dev))

naive_cor <- cor(table2_data$z_baseline, table2_data$z_naive)
overlap_cor <- cor(table2_data$z_baseline, table2_data$z_overlap)
chrono_cor <- cor(table2_data$z_baseline, table2_data$z_chrono)
aligned_cor <- cor(table2_data$z_baseline, table2_data$z_aligned)

# Table with model comparison statistics
comparison_stats <- data.frame("Model" = c("naive", "overlap", "chrono", "aligned"), 
                               "Deviance" = dev_scores, 
                               "Squared Deviance" = sq_dev_scores,
                               "Correlation" = c(naive_cor, overlap_cor, chrono_cor, aligned_cor))

writeLines(capture.output(stargazer(comparison_stats, digits = 3, summary = F, rownames = T)), 
           "./tables/table2.tex")

# One-way ANOVA (naive, overlapping, and aligned models)
anova_data <- c(table2_data$naive, table2_data$overlap, table2_data$aligned)
groups <- factor(rep(c("naive", "overlap", "aligned"), each = 35))
anova_data <- data.frame(anova_data, groups)

fit <- lm(formula = anova_data ~ groups, data = anova_data)
anova (fit)

