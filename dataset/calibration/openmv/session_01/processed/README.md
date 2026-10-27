# Dark-frame diagnostics

master-dark.png: median master dark rounded to 8-bit for display.
master-dark-display-x32.png: master dark display gain 32x.
*-corrected.png: each original dark minus the float32 master, clipped and rounded to 8-bit.
*-corrected-display-x32.png: corrected dark display gain 32x.

No blur, Top-Hat, threshold or border removal applied, so sensor edges remain inspectable.
The master includes these same 30 captures; residuals are descriptive, not independent calibration validation.
Original captures are unchanged. The .npy master retains its float32 precision.
