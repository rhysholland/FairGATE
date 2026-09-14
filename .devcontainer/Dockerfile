# Dev-container image: R + torch + tooling for DEVELOPING fairGATE.
# Your repo is mounted live; you work against it with devtools::load_all().

FROM rocker/r-ver:4.4.1

# System libraries for torch + common CRAN build deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libpng-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Provision torch's LibTorch backend at BUILD time (CPU build).
ENV TORCH_INSTALL=1
ENV TORCH_HOME=/opt/libtorch

# Dev tooling + torch.
RUN R -e "install.packages(c('remotes', 'devtools', 'torch'), repos='https://cloud.r-project.org')"

# Verify the torch backend provisioned during build.
RUN R -e "library(torch); stopifnot(torch::torch_is_installed()); cat('torch backend OK\n')"
