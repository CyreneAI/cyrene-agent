#!/bin/bash
# Script to register webhooks for Telegram and Discord
    #!/bin/bash

    # Exit immediately if a command exits with a non-zero status.
    set -e

    # IMPORTANT: Replace with your actual Telegram Bot Token
    TELEGRAM_BOT_TOKEN_1=""
    TELEGRAM_BOT_TOKEN_2=""
    # IMPORTANT: Replace with your ngrok HTTPS URL (e.g., from running 'ngrok http 8000')
    # Make sure 'ngrok http 8000' is running in a separate terminal and provides an HTTPS URL.
    NGROK_URL="https://407bf9e3cbf7.ngrok-free.app"

    WEBHOOK_URL="${NGROK_URL}/telegram/webhook"

    echo "Setting Telegram webhook to: ${WEBHOOK_URL}"
    # Use curl to call Telegram's setWebhook API
    curl "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN_1}/setWebhook?url=${WEBHOOK_URL}"
    
    echo ""

    curl "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN_2}/setWebhook?url=${WEBHOOK_URL}"

    echo ""
    echo "To get current webhook info for token 1: curl https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN_1}/getWebhookInfo"
    
    echo "To get current webhook info for token 2: curl https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN_2}/getWebhookInfo"

    echo "To delete webhook for token 1 : curl https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN_1}/deleteWebhook"

    echo "To delete webhook for token 2 : curl https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN_2}/deleteWebhook"
    