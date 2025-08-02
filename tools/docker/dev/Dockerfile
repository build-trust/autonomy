FROM nixos/nix

WORKDIR /autonomy
COPY tools/nix ./tools/nix

RUN nix-channel --update \
  && echo "experimental-features = nix-command flakes" >> /etc/nix/nix.conf \
  && nix develop ./tools/nix --command echo "Nix environment built and cached" \
  && mkdir -p /root/.local/bin \
  && mkdir -p /root/.cache

COPY . .

ENTRYPOINT ["nix", "develop", "./tools/nix", "--command"]
CMD ["bash", "-c", "source ./tools/shell/profile && exec bash"]
