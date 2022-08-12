
# This is secret and shouldn't be checked into version control
echo($WANDB_API_KEY)

# Name and notes optional
WANDB_NAME="My first run"
WANDB_NOTES="Smaller learning rate, more regularization."


python my_wandb_basic1 --log_to_wandb