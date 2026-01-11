library(reticulate)
np <- import("numpy")

breaks <- seq(0,1.2,0.1)

psd <- np$load("results/data/pool120_psd2_lsna_vrna_thrna_envna_grid3x3_dr1p0000000000000002em14_ld1p0_training.npz")
psd$files
B_psd <- psd[["B_last"]]
hist(B_psd[B_psd > 1e-2],probability = T,xlim=c(0,1),breaks = breaks)


ibm <- np$load("results/data/pool120_ibm_lsna_vrna_thrna_envna_grid3x3_dr1p0000000000000002em14_ld1p0_training.npz")
ibm$files
B_ibm <- ibm[["B_last"]]
hist(B_ibm[B_ibm > 1e-2],probability = T,col=rgb(1,0,0,0.2),add=T,border = "red", breaks = breaks)

############### Use values DUMP-ed before each iteration for more input data:
B_psd_history <- read.table("PSD2_biomass_value.txt")[,1]
B_psd_history <- B_psd_history[(length(B_psd_history)/2):length(B_psd_history)]
hist(B_psd_history,probability = T,col=rgb(0,0,1,0.2),border = "blue", breaks = breaks)

B_ibm_history <- read.table("IBM_biomass_value.txt")[,1]
B_ibm_history <- B_ibm_history[(length(B_ibm_history)/2):length(B_ibm_history)]
hist(B_ibm_history,probability = T,col=rgb(0,1,0,0.2),add=T,border = "green", breaks = breaks)

############### Use values DUMP-ed before each iteration for more input data:
B_psd_history <- read.table("PSD2_biomass_value.txt")[,1]
B_psd_history <- B_psd_history[(length(B_psd_history)/2):length(B_psd_history)]
hist(log10(B_psd_history),probability = T,col=rgb(0,0,1,0.2),border = "blue")

B_ibm_history <- read.table("IBM_biomass_value.txt")[,1]
B_ibm_history <- B_ibm_history[(length(B_ibm_history)/2):length(B_ibm_history)]
hist(log10(B_ibm_history),probability = T,col=rgb(0,1,0,0.2),add=T,border = "green")

##########################################################################
library(extraDistr)
library(pracma)

x <- extraDistr::rlgser(5000, 0.13)
mean_to_p <- function(x_bar){
  xi <- -x_bar*pracma::lambertWn(-exp(-1/x_bar)/x_bar)-1
  xi/(xi+1)
}

var_for_p <- function(p){
  -(p^2 + p*log(1-p))/((1-p)*log(1-p))^2
}

occupancy <- read.csv("occupancy.csv")
nrow(occupancy)
sum(occupancy$type==1)
mean(occupancy$type==1)
mean(occupancy$occupancy[occupancy$type==1])
p1 <- mean_to_p(mean(occupancy$occupancy[occupancy$type==1])); p1
var(occupancy$occupancy[occupancy$type==1])/var_for_p(p1)
h <- hist(occupancy$occupancy[occupancy$type==1],breaks = seq(0,2000),plot = F)
plot(h$mids,h$counts,log="y")

sum(occupancy$type==2)
mean(occupancy$type==2)
mean(occupancy$occupancy[occupancy$type==2])
p2 <- mean_to_p(mean(occupancy$occupancy[occupancy$type==2])); p2
var(occupancy$occupancy[occupancy$type==2])/var_for_p(p2)
h <- hist(occupancy$occupancy[occupancy$type==2],breaks = seq(0,2000),plot = F)
plot(h$mids,h$counts,xlim=c(1,600),log="y")

mean(occupancy$occupancy)

sum(occupancy$occupancy)

mean(occupancy$occupancy[occupancy$type==1])*sum(occupancy$type==1)
mean(occupancy$occupancy[occupancy$type==2])*sum(occupancy$type==2)
sum(occupancy$occupancy)

###########################################################################


nmax <- 50*12
cutIt <- function(n){ifelse(n > nmax, nmax, n)}
mean(cutIt(rlgser(30,theta =0.9987)))
var(cutIt(rlgser(30,theta =0.9987)))/var(rlgser(30,theta =0.9987))


