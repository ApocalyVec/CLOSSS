# Default values of arguments
REFERENCE_AUDIO_PATH='/home/apocalyvec/Downloads/yuli.caf'
SENTENCE_TO_SYNTHESIZE="My name is Eric Li and today is January 20 2021, I hope you have a great day!"

# Loop through arguments and process them
for arg in "$@"
do
    case $arg in
        -r|--reference_audio_path)
        REFERENCE_AUDIO_PATH="${arg#*=}"
        shift # Remove --initialize from processing
        ;;
        -s=*|--sentence_to_synthesize=*)
        SENTENCE_TO_SYNTHESIZE="$2"
        shift # Remove argument name from processing
        shift # Remove argument value from processing
        ;;
    esac
done

echo "# Reference audio path: $REFERENCE_AUDIO_PATH"
echo "# Sentence to synthesize: $SENTENCE_TO_SYNTHESIZE"
echo "python3 test.py -r $REFERENCE_AUDIO_PATH -sts $SENTENCE_TO_SYNTHESIZE"

python3 test.py -r "$REFERENCE_AUDIO_PATH" -sts "$SENTENCE_TO_SYNTHESIZE"
