## DISCLAIMER

This was 100% vibe-coded using Claude. That means I know exactly what techniques and design decisions were made, but I barely understand any of the code. That also means if something doesn't work, your best bet is to just throw all of it into an LLM and ask it to fix it. It also means you probably shouldn't use this for anything important or expose it to anyone. You have been warned.

## Description

OnePane is a simple utility that solves a recurring problem; you've run go-witness or aquatone or whatever your favorite screen-grabing tool might be, but now you have too many screenshots to look through manually, and you JUST KNOW that a ton of them are gonna be identical. *SIGH*!

Fear not, OnePane got your back.

This tool groups similar screenshots using perceptual hashing and computer vision techniques. It performs edge-based structure detection to identify similar layouts, performs color layout vector analysis to identify similar color themes, measures the Hamming distance of perceptual hashes, wavelet and blurs text before comparison so that `hostname1` and `hostname2` doesnt' result in screenshots not being grouped.

It supports two detection modes:
1. Exact duplicates: Near-identical images (default threshold: 97%)
2. Structural similarity: Same layout/UI with different text (default threshold: 97%)

You can tweak the thresholds, but in testing I found that 97% appears to be the magic number that doesn't result in too many single-item groups and a fairly low number of false positives (screenshot being added to group despite being completely different), but try it out.

The script generates a web interface for browsing grouped screenshots, which you can scale up and down to accommodate your monitor size. Each row of grouped screenshots come with an auto-scroll feature, and you can even click the master scroll-button at the top to just start scrolling through everything automatically, one row at a time (with an adjustable scrolling speed).

Parsing and grouping the screenshots is fairly quick, and in testing, 5000 screenshots were processed, grouped and displayed in approx. 15 minutes.
Results are heavily cached for performance across multiple runs, so unless you change the comparison thresholds or add new pictures, the web interface loads in seconds. If this is something you keep coming back to, you can also click the group names and change them to something meaningful, which will likewise persist across runs.

Speaking of run:

## Getting started
```bash
Usage: python3 script.py <folder_path> [exact_threshold] [structural_threshold] [port]
  folder_path: Directory containing screenshots
  exact_threshold: Exact duplicate threshold (default: 97)
  structural_threshold: Structural similarity threshold (default: 97)
  port: Web server port (default: 5000)
```

### Example

```bash
python3 OnePane.py /path/to/folder/with/screenshots/
```

This will run a Flask web application on your localhost port 5000, using the default thresholds

<img width="1685" height="585" alt="image" src="https://github.com/user-attachments/assets/0ab00ee6-39f4-45c8-8fe2-d34656e6bde9" />

### Suggested workflow

Run your favourite portscanner. For big networks where you can afford to miss a system or two in return for getting wide coverage, something like https://github.com/robertdavidgraham/masscan works really well, and can be run like so:
```bash
masscan -p80,8000-8100 10.0.0.0/8 --max-rate 10000 -oG outputfile
```
Just make sure to include all the ports where you expect there would be web interfaces on (and that you know how to attack), then run your favourite screen-grabbing tool, like https://github.com/michenriksen/aquatone (which inspired this tool), or https://github.com/sensepost/gowitness:
```bash
gowitness scan -nmap -f outputfile --threads 50 --open-only --write-jsonl outputfile --write-screenshots
```
Let it run, and now point OnePane at the screenshots folder:
```bash
python3 OnePane.py screenshots/ 97 97 5000
```
Browse to http://localhost:5000, sit back and enjoy the show!
