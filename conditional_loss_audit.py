"""Run a held-out conditional denoising audit for text conditioning.

This script evaluates whether the checkpoint assigns lower denoising loss to
the correct prompt than to shuffled or null prompts on real validation clips.
It also computes an in-batch prompt-ranking accuracy by comparing the loss of
each real clip under each prompt in the batch.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import get_dataloader
from trainer import create_trainer


def _bootstrap_ci(values: List[float], seed: int, num_bootstrap: int = 2000) -> Dict[str, float]:
	arr = np.asarray(values, dtype=np.float64)
	if arr.size == 0:
		return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
	rng = np.random.default_rng(seed)
	means = []
	for _ in range(num_bootstrap):
		sample = rng.choice(arr, size=arr.size, replace=True)
		means.append(float(sample.mean()))
	means_arr = np.asarray(means)
	return {
		"mean": float(arr.mean()),
		"ci95_low": float(np.percentile(means_arr, 2.5)),
		"ci95_high": float(np.percentile(means_arr, 97.5)),
	}


def _encode_tokens(trainer, tokens: torch.Tensor) -> torch.Tensor:
	return trainer.text_encoder(tokens=tokens.to(trainer.device), text=None)


def _sample_derangement(size: int, device: torch.device) -> torch.Tensor:
	if size <= 1:
		return torch.arange(size, device=device)
	perm = torch.randperm(size, device=device)
	while torch.any(perm == torch.arange(size, device=device)):
		perm = torch.randperm(size, device=device)
	return perm


def _normalize_words(text: str) -> set[str]:
	return set(re.findall(r"[a-z0-9']+", text.lower()))


def _jaccard_overlap(text_a: str, text_b: str) -> float:
	words_a = _normalize_words(text_a)
	words_b = _normalize_words(text_b)
	if not words_a and not words_b:
		return 0.0
	return len(words_a & words_b) / max(len(words_a | words_b), 1)


def _build_batches(samples: List[Dict], batch_size: int, diverse_batches: bool) -> List[List[Dict]]:
	if not diverse_batches or batch_size <= 1:
		return [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]

	unused = list(samples)
	batches: List[List[Dict]] = []
	while unused:
		current_batch = [unused.pop(0)]
		while unused and len(current_batch) < batch_size:
			best_idx = 0
			best_score = float("inf")
			for idx, candidate in enumerate(unused):
				overlaps = [_jaccard_overlap(candidate["text"], existing["text"]) for existing in current_batch]
				score = max(overlaps) if overlaps else 0.0
				if score < best_score:
					best_score = score
					best_idx = idx
			current_batch.append(unused.pop(best_idx))
		batches.append(current_batch)
	return batches


@torch.no_grad()
def _samplewise_loss(trainer, video, timesteps, noise, text_embeddings) -> torch.Tensor:
	noisy_video = trainer.scheduler.add_noise(video, noise, timesteps)
	with trainer._autocast_context():
		noise_pred = trainer.model(noisy_video, timesteps, text_embeddings)

	if trainer.ddim_config.prediction_type == "epsilon":
		target = noise
	else:
		target = trainer.scheduler.get_velocity(video, noise, timesteps)

	loss = F.mse_loss(noise_pred, target, reduction="none")
	return loss.mean(dim=(1, 2, 3, 4)).float().detach().cpu()


def main() -> None:
	parser = argparse.ArgumentParser(description="Run held-out conditional denoising audit")
	parser.add_argument("--checkpoint", required=True, help="Path to best checkpoint")
	parser.add_argument("--data-dir", required=True, help="Path to training/validation data root")
	parser.add_argument("--output", required=True, help="Output JSON path")
	parser.add_argument("--num-samples", type=int, default=24, help="Number of validation samples")
	parser.add_argument("--batch-size", type=int, default=4, help="Validation batch size for audit")
	parser.add_argument("--repeats", type=int, default=3, help="Independent noise/timestep repeats per batch")
	parser.add_argument("--max-word-count", type=int, default=None, help="Optional max word count filter")
	parser.add_argument("--unique-prompts", action="store_true", help="Require unique prompt strings in the selected audit set")
	parser.add_argument("--diverse-batches", action="store_true", help="Greedily group prompts into lexically diverse batches")
	parser.add_argument("--min-timestep", type=int, default=0, help="Minimum diffusion timestep (inclusive)")
	parser.add_argument("--max-timestep", type=int, default=None, help="Maximum diffusion timestep (exclusive)")
	parser.add_argument("--seed", type=int, default=1234, help="Random seed")
	parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation")
	args = parser.parse_args()

	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	checkpoint_path = Path(args.checkpoint)
	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	checkpoint = torch.load(checkpoint_path, map_location="cpu")
	model_config = checkpoint["model_config"]
	train_config = checkpoint["train_config"]
	ddim_config = checkpoint["ddim_config"]

	train_config.data_dir = args.data_dir
	train_config.device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
	train_config.enable_compile = False
	train_config.use_amp = not args.cpu and torch.cuda.is_available()
	train_config.precision = "auto" if train_config.use_amp else "fp32"
	train_config.num_workers = 0
	train_config.dataloader_pin_memory = False
	train_config.dataloader_persistent_workers = False
	train_config.dataloader_prefetch_factor = None
	train_config.batch_size = args.batch_size

	trainer = create_trainer(model_config, train_config, ddim_config, run_name="conditional_loss_audit")
	trainer.load_checkpoint(str(checkpoint_path), resume_training=True)
	trainer.model.eval()
	trainer.text_encoder.eval()

	max_timestep = args.max_timestep if args.max_timestep is not None else trainer.ddim_config.num_train_timesteps
	if not (0 <= args.min_timestep < max_timestep <= trainer.ddim_config.num_train_timesteps):
		raise ValueError(
			f"Invalid timestep range [{args.min_timestep}, {max_timestep}) for num_train_timesteps={trainer.ddim_config.num_train_timesteps}"
		)

	selection_loader = get_dataloader(
		data_dir=train_config.data_dir,
		batch_size=1,
		image_size=model_config.image_size,
		num_frames=model_config.num_frames,
		num_workers=0,
		train=False,
		train_ratio=train_config.train_ratio,
		split_mode=train_config.split_mode,
		random_seed=train_config.split_seed,
		tokenizer=trainer.text_encoder.tokenizer if hasattr(trainer.text_encoder, "tokenizer") else trainer.tokenizer,
		use_length_prefix=getattr(model_config, "use_length_prefix", False),
		pin_memory=False,
		persistent_workers=False,
		prefetch_factor=None,
	)

	selected_samples = []
	seen_prompts = set()
	for batch in selection_loader:
		text = batch["text"][0]
		word_count = len(text.replace("_", " ").split())
		if args.max_word_count is not None and word_count > args.max_word_count:
			continue
		if args.unique_prompts and text in seen_prompts:
			continue
		selected_samples.append(
			{
				"video": batch["video"][0],
				"tokens": batch["tokens"][0],
				"text": text,
				"word_count": word_count,
			}
		)
		seen_prompts.add(text)
		if len(selected_samples) >= args.num_samples:
			break

	if not selected_samples:
		raise RuntimeError("No samples matched the requested audit filters.")

	batches = _build_batches(selected_samples, args.batch_size, args.diverse_batches)

	normal_losses: List[float] = []
	none_losses: List[float] = []
	random_losses: List[float] = []
	normal_minus_none: List[float] = []
	normal_minus_random: List[float] = []
	top1_hits: List[float] = []
	top3_hits: List[float] = []
	correct_ranks: List[float] = []
	best_wrong_minus_correct: List[float] = []
	batch_records = []

	samples_processed = 0

	try:
		if trainer.ema is not None:
			trainer.ema.apply_shadow()

		max_batches = len(batches)
		for batch_idx, batch_samples in enumerate(tqdm(batches, total=max_batches, desc="Conditional loss audit")):
			if not batch_samples:
				break

			video = torch.stack([sample["video"] for sample in batch_samples]).to(trainer.device).permute(0, 2, 1, 3, 4)
			texts = [sample["text"] for sample in batch_samples]
			batch_tokens = torch.stack([sample["tokens"] for sample in batch_samples]).to(trainer.device)
			batch_size = video.shape[0]

			normal_embeddings = _encode_tokens(trainer, batch_tokens)
			none_embeddings = torch.zeros_like(normal_embeddings)

			repeat_normal = []
			repeat_none = []
			repeat_random = []
			repeat_matrix = []

			for repeat_idx in range(args.repeats):
				timesteps = torch.randint(
					args.min_timestep,
					max_timestep,
					(batch_size,),
					device=trainer.device,
				)
				noise = torch.randn_like(video)

				random_perm = _sample_derangement(batch_size, trainer.device)
				random_embeddings = normal_embeddings[random_perm]

				loss_normal = _samplewise_loss(trainer, video, timesteps, noise, normal_embeddings)
				loss_none = _samplewise_loss(trainer, video, timesteps, noise, none_embeddings)
				loss_random = _samplewise_loss(trainer, video, timesteps, noise, random_embeddings)

				repeat_normal.append(loss_normal.numpy())
				repeat_none.append(loss_none.numpy())
				repeat_random.append(loss_random.numpy())

				loss_matrix = np.zeros((batch_size, batch_size), dtype=np.float32)
				for prompt_idx in range(batch_size):
					prompt_tokens = batch_tokens[prompt_idx:prompt_idx + 1].expand(batch_size, -1)
					prompt_embeddings = _encode_tokens(trainer, prompt_tokens)
					loss_matrix[:, prompt_idx] = _samplewise_loss(
						trainer, video, timesteps, noise, prompt_embeddings
					).numpy()
				repeat_matrix.append(loss_matrix)

			mean_normal = np.mean(np.stack(repeat_normal, axis=0), axis=0)
			mean_none = np.mean(np.stack(repeat_none, axis=0), axis=0)
			mean_random = np.mean(np.stack(repeat_random, axis=0), axis=0)
			mean_matrix = np.mean(np.stack(repeat_matrix, axis=0), axis=0)

			for sample_idx in range(batch_size):
				row = mean_matrix[sample_idx]
				order = np.argsort(row)
				correct_rank = int(np.where(order == sample_idx)[0][0]) + 1
				best_wrong = float(np.min(np.delete(row, sample_idx))) if batch_size > 1 else float(row[sample_idx])
				correct_loss = float(row[sample_idx])

				normal_losses.append(float(mean_normal[sample_idx]))
				none_losses.append(float(mean_none[sample_idx]))
				random_losses.append(float(mean_random[sample_idx]))
				normal_minus_none.append(float(mean_none[sample_idx] - mean_normal[sample_idx]))
				normal_minus_random.append(float(mean_random[sample_idx] - mean_normal[sample_idx]))
				top1_hits.append(float(order[0] == sample_idx))
				top3_hits.append(float(correct_rank <= min(3, batch_size)))
				correct_ranks.append(float(correct_rank))
				best_wrong_minus_correct.append(float(best_wrong - correct_loss))

				batch_records.append(
					{
						"batch_index": batch_idx,
						"sample_index": sample_idx,
						"text": texts[sample_idx],
						"normal_loss": float(mean_normal[sample_idx]),
						"none_loss": float(mean_none[sample_idx]),
						"random_loss": float(mean_random[sample_idx]),
						"correct_rank": correct_rank,
						"top1_hit": bool(order[0] == sample_idx),
						"top3_hit": bool(correct_rank <= min(3, batch_size)),
						"best_wrong_minus_correct": float(best_wrong - correct_loss),
					}
				)

			samples_processed += batch_size
			if samples_processed >= args.num_samples:
				break
	finally:
		if trainer.ema is not None:
			trainer.ema.restore()

	summary = {
		"checkpoint": str(checkpoint_path),
		"num_samples": int(min(samples_processed, args.num_samples)),
		"batch_size": args.batch_size,
		"repeats": args.repeats,
		"selection": {
			"max_word_count": args.max_word_count,
			"unique_prompts": bool(args.unique_prompts),
			"diverse_batches": bool(args.diverse_batches),
			"min_timestep": args.min_timestep,
			"max_timestep": max_timestep,
		},
		"device": train_config.device,
		"normal_loss": _bootstrap_ci(normal_losses, args.seed + 1),
		"none_loss": _bootstrap_ci(none_losses, args.seed + 2),
		"random_loss": _bootstrap_ci(random_losses, args.seed + 3),
		"none_minus_normal": _bootstrap_ci(normal_minus_none, args.seed + 4),
		"random_minus_normal": _bootstrap_ci(normal_minus_random, args.seed + 5),
		"prompt_ranking": {
			"top1_accuracy": _bootstrap_ci(top1_hits, args.seed + 6),
			"top3_accuracy": _bootstrap_ci(top3_hits, args.seed + 7),
			"mean_correct_rank": _bootstrap_ci(correct_ranks, args.seed + 8),
			"mean_best_wrong_minus_correct": _bootstrap_ci(best_wrong_minus_correct, args.seed + 9),
			"chance_top1": 1.0 / max(args.batch_size, 1),
		},
		"records": batch_records,
	}

	output_path.write_text(json.dumps(summary, indent=2))
	print(f"Saved conditional-loss audit to {output_path}")
	print(json.dumps(summary, indent=2))


if __name__ == "__main__":
	main()
