#!/bin/bash

# Google Drive file downloader
# File ID extracted from the shared link
FILE_ID="1nPmeVhkda8rMXT6s-cXmlWD0L9lzzeLz"
OUTPUT_FILE="ctnexus_eao.tar.gz"

echo "Downloading file from Google Drive..."

# First attempt - direct download (works for small files)
CONFIRM_URL="https://drive.usercontent.google.com/download?id=${FILE_ID}&export=download"

# Use curl with redirect following and cookie handling (needed for large file virus scan warning)
curl -L -o "${OUTPUT_FILE}" \
  -H "User-Agent: Mozilla/5.0" \
  --cookie-jar /tmp/gdrive_cookies \
  --cookie /tmp/gdrive_cookies \
  "${CONFIRM_URL}"

# Check if we got an HTML page (large file confirmation page) instead of the actual file
if file "${OUTPUT_FILE}" | grep -q "HTML"; then
    echo "Large file detected — handling confirmation prompt..."

    # Extract the confirmation token and retry
    CONFIRM_TOKEN=$(grep -o 'confirm=[^&]*' /tmp/gdrive_cookies | sed 's/confirm=//')

    curl -L -o "${OUTPUT_FILE}" \
      -H "User-Agent: Mozilla/5.0" \
      --cookie-jar /tmp/gdrive_cookies \
      --cookie /tmp/gdrive_cookies \
      "${CONFIRM_URL}&confirm=t"
fi

# Clean up cookies
rm -f /tmp/gdrive_cookies

# Show result
if [ -f "${OUTPUT_FILE}" ]; then
    FILE_SIZE=$(du -h "${OUTPUT_FILE}" | cut -f1)
    echo "Download complete: ${OUTPUT_FILE} (${FILE_SIZE})"
else
    echo "Download failed."
    exit 1
fi