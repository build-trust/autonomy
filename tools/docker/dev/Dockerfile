FROM nixos/nix

WORKDIR /autonomy
COPY tools/nix ./tools/nix

RUN echo "experimental-features = nix-command flakes" >> /etc/nix/nix.conf \
  && echo "sandbox = relaxed" >> /etc/nix/nix.conf \
  && echo "filter-syscalls = false" >> /etc/nix/nix.conf \
  && echo "download-buffer-size = 268435456" >> /etc/nix/nix.conf \
  && nix-channel --update \
  && nix develop ./tools/nix --command echo "Nix environment built and cached" \
  && mkdir -p /root/.local/bin \
  && mkdir -p /root/.cache

COPY . .

ENTRYPOINT ["nix", "develop", "./tools/nix", "--command"]
CMD ["bash", "-c", "source ./tools/shell/profile && exec bash"]
