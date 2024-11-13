from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score

def evaluate(model, dataloader, tokenizer):
    model.eval()
    references, hypotheses = [], []

    with torch.no_grad():
        for images, captions in dataloader:
            outputs = model(images)
            predictions = tokenizer.batch_decode(outputs.argmax(dim=-1), skip_special_tokens=True)

            for ref, hyp in zip(captions, predictions):
                references.append([ref])
                hypotheses.append(hyp)

    # Compute BLEU and METEOR
    bleu = sum([sentence_bleu(ref, hyp) for ref, hyp in zip(references, hypotheses)]) / len(references)
    meteor = sum([meteor_score(ref, hyp) for ref, hyp in zip(references, hypotheses)]) / len(references)
    print(f"BLEU: {bleu:.4f}, METEOR: {meteor:.4f}")
