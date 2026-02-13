# Cutout Detection Improvements

## Changes Made

### 1. **Confidence Levels**
Cutouts now have three confidence levels:
- **Definite**: Clear cutouts (silence gaps, very high grammar issues)
- **Likely**: Probable cutouts (moderate grammar issues, empty acknowledgments)
- **Maybe**: Possible cutouts (low grammar issues)

### 2. **Improved Silence Detection**
- **Before**: Flagged all unusual silence gaps
- **Now**: Only flags silence when AGENT speaks → long gap → HUMAN responds
- **Why**: When HUMAN speaks and takes time to respond, that's normal thinking time, not a cutout

### 3. **Grammar Issue Severity**
Grammar issues are now classified as:
- **High**: Z-score > 3.5 → "Definite" confidence
- **Medium**: Z-score > 2.5 → "Likely" confidence  
- **Low**: Z-score > threshold but < 2.5 → "Maybe" confidence

## Output Changes

### CSV Output
New column: `cutout_confidence` (definite/likely/maybe)

### JSON Report
Each cutout now includes:
```json
{
  "start": 10.5,
  "end": 12.3,
  "text": "Example text",
  "detected_issues": ["Grammar Issue - Low (Score 2.3)"],
  "confidence": "maybe"
}
```

### Console Output
```
📊 RESULTS:
   Turns: 25
   Total Cutouts: 8
   - Definite: 2
   - Likely: 3
   - Maybe: 3
   Avg Latency: 1.23s
   Total Cutout Duration: 15.67s
```

## Usage Example

After getting results, you can filter by confidence:
- Review **Definite** cutouts first (high priority)
- Check **Likely** cutouts (medium priority)
- Optionally review **Maybe** cutouts (low priority, may be false positives)
