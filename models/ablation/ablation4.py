    def add_gaussian_noise(self, x, noise_std=0.1):
        if noise_std <= 0:
            return x

        noise = torch.randn_like(x) * noise_std
        return x + noise

    def patch_mask(self, x, mask_ratio=None, mask_value=0.0):
        b, c, h, w = x.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(
                f"Input size ({h}, {w}) must be divisible by patch_size ({self.patch_size})."
            )

        if mask_ratio is None:
            mask_ratio = self.p
        mask_ratio = max(0.0, min(1.0, float(mask_ratio)))

        gh = h // self.patch_size
        gw = w // self.patch_size
        n = gh * gw
        num_to_mask = int(mask_ratio * n)

        if num_to_mask <= 0:
            return x

        patches = x.view(b, c, gh, self.patch_size, gw, self.patch_size)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(
            b, n, c, self.patch_size, self.patch_size
        )

        rand = torch.rand(b, n, device=x.device)
        mask_idx = torch.argsort(rand, dim=1)[:, :num_to_mask]
        patch_mask = torch.zeros(b, n, dtype=torch.bool, device=x.device)
        patch_mask.scatter_(1, mask_idx, True)

        masked = patches.masked_fill(
            patch_mask[:, :, None, None, None],
            mask_value,
        )

        masked = masked.view(b, gh, gw, c, self.patch_size, self.patch_size)
        masked = masked.permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)
        return masked
