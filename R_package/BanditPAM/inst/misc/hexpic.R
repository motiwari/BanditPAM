library(tidyverse)
library(BanditPAM)
library(hexSticker)

set.seed(12293)
n_per_cluster <- 20
means <- list(c(0, 0), c(-5, 5), c(5, 5))

X <- do.call(rbind, lapply(means, MASS::mvrnorm, n = n_per_cluster, Sigma = diag(2)))
obj <- KMedoids$new(k = 3)
obj$fit(data = X, loss = "l2")
med_indices <- obj$get_medoids_final()

d <- cbind(as_tibble(X),
           tibble(color = c(rep("bisque", n_per_cluster),
                            rep("cyan4", n_per_cluster),
                            rep("darkorange", n_per_cluster)))
           )
names(d) <- c("x", "y", "color")
medoids <- d[med_indices, ] ## just the medoids
d <- d[-med_indices, ] ## the rest

ggplot(data = d) +
  geom_point(aes(x, y, color = color), show.legend = FALSE, size = 1.5) +
  geom_point(aes(x, y, fill = color), shape = 21, color = "darkblue",
             show.legend = FALSE, size = 1.5, data = medoids) +
  theme_void() ->
  p


sticker(p, package = "BanditPAM", p_color = "aliceblue", p_family = "sans", p_fontface = "italic",
        p_size = 16, s_x = 1.0, s_y = 0.75, s_width = 1.7, s_height = 1.3,
        h_fill = "deepskyblue4", h_size = 0.75, h_color = "darkgoldenrod1", filename = "banditpam.png")

