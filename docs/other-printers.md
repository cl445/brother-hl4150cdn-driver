# Other Brother printers

Brother wrote its Linux drivers for a whole range of colour lasers in the
same way: a 32-bit x86 filter that turns a page into PJL plus an XL2HB
raster stream. This driver reproduces the HL-4150CDN's filter. Many other
models use the same filter, or one built from the same code, so they could
be supported too.

This page lists what is known about them. It is based on Brother's Linux
packages (LPR and cupswrapper `.deb` files from `download.brother.com`) for
the models below: their filters were compared with the HL-4150CDN's and
run in Docker on the same test page. Nothing here has been tested on paper
except the HL-4150CDN itself.

Do you have one of these printers? See [How you can help](#how-you-can-help).

## Groups

| Group | Models | What differs from the HL-4150CDN | Effort |
|---|---|---|---|
| **Same filter** | HL-4140CN, HL-4570CDW, DCP-9055CDN, DCP-9270CDN, MFC-9460CDN, MFC-9465CDN, MFC-9560CDW, MFC-9970CDW | Nothing: the same filter binary and the same tables (LPR driver 1.1.1-5). Brother's filter gives the MFC-9970CDW byte for byte the same output as the HL-4150CDN, in Normal and Fine. The HL-4140CN has no duplex unit. | Testers wanted; then a PPD name per model |
| **Same code, other data** | HL-3040CN, HL-3045CN, HL-3070CW, HL-3075CW, DCP-9010CN, MFC-9010CN, MFC-9120CN, MFC-9125CN, MFC-9320CW, MFC-9325CW | Colour and dither tables (`cache08`), page geometry, a PJL job block, a gray tone curve. The encoders are identical. A port for the HL-3040CN exists: [pedroterzero/brother-hl3040cn-driver](https://github.com/pedroterzero/brother-hl3040cn-driver). | A second `PrinterModel`, a few days |
| **Same codecs, other colour stage** | HL-3140CW, HL-3150CDN, HL-3150CDW, HL-3170CDW, HL-3152CDW, HL-3172CDW, HL-3180CDW, DCP-9015CDW, DCP-9017CDW, DCP-9020CDN, DCP-9020CDW, MFC-9130CW, MFC-9140CDN, MFC-9142CDN, MFC-9330CDW, MFC-9332CDW, MFC-9340CDW, MFC-9342CDW | Normal mode encodes K and Y with the 10-bit encoder this driver uses for M. Grayscale in Fine mode is identical. The colour separation differs and has to be worked out from the binary. | About 1–2 weeks of reverse engineering |
| **New encoders** | HL-L8250CDN, HL-L8350CDW, HL-L9200CDW, DCP-L8400CDN, DCP-L8450CDW, MFC-L8600CDW, MFC-L8650CDW, MFC-L8850CDW, MFC-L9550CDW | 8-bit and 17-bit Normal-mode encoders, a different image height, a different compiler (no shared code to diff against). | Weeks |
| | HL-L3210CW, HL-L3230CDW, HL-L8260CDW, HL-L8360CDW (likely also MFC-L3770CDW, HL-L9310CDW, MFC-L8900CDW) | A new Normal-mode encoder type. Grayscale in Fine mode is identical. | Weeks |
| | HL-4040CN, HL-4040CDN, HL-4050CDN, HL-4070CDW, DCP-9040CN, DCP-9042CDN, DCP-9045CDN, MFC-9440CN, MFC-9450CDN, MFC-9840CDW | An older filter: another K/Y encoder, different PJL, a different Fine mode switch. | Weeks |
| **Not XL2HB** | HL-L3280CDW, HL-L8230CDW, HL-L8240CDW, HL-L3220CW | Brother's driver sends PWG Raster, an open format. CUPS' own drivers (driverless/IPP Everywhere) are the better route. | Not needed |

Facts that hold for every XL2HB family above:

- Fine mode (600x2400 dpi) uses the same codec and band layout.
- The colour matching *None* grids and the interpolation tables are the
  same in all of them.
- In Normal mode, the encoder per colour plane is announced in the
  BeginImage attribute `0x81`. This driver writes the HL-4150CDN's choice;
  a model with other encoders needs it as a model setting.

## How you can help

**You have a printer from the "Same filter" group.** It very likely works
with this driver as it is. Install it as described in the
[README](../README.md#install) and register the queue with the HL-4150CDN
PPD, then print a few pages (colour, grayscale, duplex if your printer has
it, Fine quality) and
[open a test report](https://github.com/cl445/brother-hl4150cdn-driver/issues/new?template=test-report.yml).
A report that it works is just as useful as one that it does not.

**You have a printer from another group and want to add it.**
[Adding a printer model](adding-a-model.md) describes the work. Please
[open a printer support issue](https://github.com/cl445/brother-hl4150cdn-driver/issues/new?template=printer-support.yml)
first, or comment on the [existing one for your family](https://github.com/cl445/brother-hl4150cdn-driver/issues?q=is%3Aissue+label%3Aprinter-support),
so work is not done twice. The maintainer has no access to these printers
and will not add them alone, but reviews pull requests and helps with the
tooling.

## Open questions

- The HL-3040CN's filter switches to an 8-bit encoder on all planes at a
  line width of 4760 pixels. Which rule picks the encoder, and does the
  HL-4150CDN's filter have it too?
- What exactly changed in the colour stage from the HL-3150CDN family on?
- Why does the HL-4050CDN family's Fine grayscale output differ even with
  the same tables?
