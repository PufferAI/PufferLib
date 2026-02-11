#!/usr/bin/env python
"""League System Orchestrator for Dogfight.

Manages a population of policies through evaluate → train → verify → promote/reject
rounds. Policies are collected from W&B sweeps and improved through league training
with PFSP opponent selection.

Usage:
    # Round 0: Collect from W&B + initial eval
    python pufferlib/ocean/dogfight/league.py collect \
        --project df27 --top-n 20 --output-dir league/

    python pufferlib/ocean/dogfight/league.py eval \
        --manifest league/manifest.json --games-per-pair 30

    # Round 1+: Full round (eval → train → verify → promote/reject → re-eval)
    python pufferlib/ocean/dogfight/league.py round \
        --manifest league/manifest.json \
        --steps 50000000 --games-per-pair 30 --games-per-ref 100

    # Individual phases
    python pufferlib/ocean/dogfight/league.py train \
        --manifest league/manifest.json --steps 50000000

    python pufferlib/ocean/dogfight/league.py verify \
        --manifest league/manifest.json --games-per-ref 100

    # Status
    python pufferlib/ocean/dogfight/league.py status \
        --manifest league/manifest.json
"""
import argparse
import os
import shutil
import time
from datetime import datetime

import torch


class League:
    def __init__(self, manifest_path, device='cuda'):
        from pufferlib.ocean.dogfight.league_manifest import LeagueManifest
        self.manifest_path = manifest_path
        self.league_dir = os.path.dirname(manifest_path)
        self.device = device
        self.manifest = LeagueManifest.load(manifest_path)

    def evaluate(self, games_per_pair=30, num_envs=64, num_workers=None):
        """Run round-robin tournament, update manifest with ratings."""
        from pufferlib.ocean.dogfight.elo_eval import run_league_tournament

        all_policies = self.manifest.policies
        print(f'[LEAGUE] Evaluating {len(all_policies)} policies '
              f'({games_per_pair} games/pair, {num_envs} parallel envs)')

        win_matrix, ratings = run_league_tournament(
            all_policies, self.league_dir,
            games_per_pair=games_per_pair,
            num_envs=num_envs,
            device=self.device,
            num_workers=num_workers,
        )

        # Update policy ratings
        for p in self.manifest.policies:
            if p.id in ratings:
                p.rating = ratings[p.id]

        # Record round
        from pufferlib.ocean.dogfight.league_manifest import RoundRecord
        round_num = self.manifest.next_round_number()
        record = RoundRecord(
            round=round_num,
            type='initial_eval' if round_num == 0 else 'eval',
            timestamp=datetime.now().isoformat(),
            win_matrix=win_matrix,
            ratings=ratings,
        )
        self.manifest.add_round(record)

        # Update rating history
        for p in self.manifest.policies:
            if p.id in ratings:
                self.manifest.update_rating_history(
                    p.id, round_num, ratings[p.id], p.generation)

        self.manifest.save(self.manifest_path)
        print(f'[LEAGUE] Evaluation complete (round {round_num})')
        return win_matrix, ratings

    def train_all(self, steps_per_policy=50_000_000, wandb_project=None):
        """Train ALL active policies sequentially.

        Returns dict: policy_id -> candidate_checkpoint_path
        """
        from pufferlib.ocean.dogfight.train_dual_selfplay import train_league_round

        active = self.manifest.get_active_policies()
        candidates = {}

        print(f'[LEAGUE] Training {len(active)} active policies '
              f'({steps_per_policy} steps each)')

        for i, policy in enumerate(active):
            if policy.flagged_for_review:
                print(f'[LEAGUE] Skipping {policy.id} (flagged for review)')
                continue

            # Check for existing candidate (crash recovery)
            candidate_dir = os.path.join(self.league_dir, 'candidates')
            next_gen = policy.generation + 1
            candidate_filename = f'{policy.id}_gen{next_gen}_candidate.pt'
            candidate_path = os.path.join(candidate_dir, candidate_filename)

            if os.path.exists(candidate_path):
                print(f'[LEAGUE] Found existing candidate for {policy.id}, skipping training')
                candidates[policy.id] = candidate_path
                continue

            print(f'\n[LEAGUE] Training policy {i+1}/{len(active)}: {policy.id}')
            start_time = time.time()

            try:
                candidate_path = train_league_round(
                    policy, self.manifest, self.league_dir,
                    training_steps=steps_per_policy,
                    device=self.device,
                    wandb_project=wandb_project,
                )
                if candidate_path:
                    candidates[policy.id] = candidate_path
                    elapsed = time.time() - start_time
                    print(f'[LEAGUE] {policy.id} trained in {elapsed:.0f}s')
            except Exception as e:
                print(f'[LEAGUE] ERROR training {policy.id}: {e}')
                import traceback
                traceback.print_exc()

            # Save manifest after each policy (crash recovery)
            self.manifest.save(self.manifest_path)

        print(f'[LEAGUE] Training complete: {len(candidates)}/{len(active)} candidates')
        return candidates

    def verify(self, candidates, games_per_ref=100, num_envs=64):
        """Run verification gauntlet for each candidate.

        Args:
            candidates: dict policy_id -> candidate_checkpoint_path
            games_per_ref: Games per reference opponent (50 from each side).
            num_envs: Number of parallel environments.

        Returns:
            dict: policy_id -> verification_result dict
        """
        from pufferlib.ocean.dogfight.elo_eval import (
            load_policy_from_path, run_matches, compute_elo_mle
        )
        from pufferlib.ocean.dogfight.dogfight import Dogfight
        from pufferlib.ocean.dogfight import binding

        results = {}

        for policy_id, candidate_path in candidates.items():
            policy = self.manifest.get_policy_by_id(policy_id)
            if policy is None:
                continue

            print(f'\n[VERIFY] Gauntlet for {policy_id} (gen {policy.generation + 1})')

            # Build reference set: same-scheme anchors + own best checkpoint
            references = []

            # All frozen anchors in same scheme
            same_scheme = self.manifest.get_policies_by_scheme(policy.obs_scheme)
            for ref in same_scheme:
                if ref.status == 'frozen':
                    ref_path = os.path.join(self.league_dir, ref.model_path)
                    if os.path.exists(ref_path):
                        references.append({
                            'id': ref.id,
                            'path': ref_path,
                            'hidden_size': ref.hidden_size,
                        })

            # Own best checkpoint
            best_path = os.path.join(self.league_dir, policy.best_checkpoint_path)
            if os.path.exists(best_path):
                references.append({
                    'id': f'own_best_gen{policy.best_checkpoint_generation}',
                    'path': best_path,
                    'hidden_size': policy.hidden_size,
                })

            if not references:
                print(f'[VERIFY] No references found for {policy_id}, auto-promoting')
                results[policy_id] = {
                    'decision': 'PROMOTED',
                    'reason': 'No references available',
                    'gauntlet_results': {},
                }
                continue

            # Create env for this scheme
            env = Dogfight(
                num_envs=1,
                render_mode=None,
                obs_scheme=policy.obs_scheme,
                curriculum_enabled=1,
                fixed_stage=20,
                max_steps=6000,
            )
            binding.vec_enable_opponent_override(env.c_envs, 1)

            # Load candidate
            candidate_policy = load_policy_from_path(
                candidate_path, env, self.device,
                hidden_size=policy.hidden_size)

            # Run gauntlet
            gauntlet = {}
            matchup_results = []
            ref_elos = {}

            for ref in references:
                ref_policy = load_policy_from_path(
                    ref['path'], env, self.device,
                    hidden_size=ref['hidden_size'])

                # Play from both sides to cancel spawn position asymmetry
                half = games_per_ref // 2
                r1 = run_matches(env, candidate_policy, ref_policy,
                                 half, self.device)
                r2 = run_matches(env, ref_policy, candidate_policy,
                                 games_per_ref - half, self.device)
                # Combine: r1 wins are candidate wins; r2 losses are candidate wins
                result = {
                    'wins': r1['wins'] + r2['losses'],
                    'losses': r1['losses'] + r2['wins'],
                    'draws': r1['draws'] + r2['draws'],
                }

                total = result['wins'] + result['losses'] + result['draws']
                win_rate = result['wins'] / max(total, 1)

                gauntlet[ref['id']] = {
                    'win_rate': win_rate,
                    'games': total,
                    'wins': result['wins'],
                    'losses': result['losses'],
                    'draws': result['draws'],
                }

                # For rating computation
                ref_elo = 1000.0  # Default
                ref_policy_entry = self.manifest.get_policy_by_id(ref['id'])
                if ref_policy_entry:
                    ref_elo = ref_policy_entry.rating
                ref_elos[ref['id']] = ref_elo
                matchup_results.append({
                    'opponent_tag': ref['id'],
                    'wins': result['wins'],
                    'losses': result['losses'],
                    'draws': result['draws'],
                })

                print(f'  vs {ref["id"][:40]}: {result["wins"]}W/{result["losses"]}L/{result["draws"]}D '
                      f'({win_rate:.0%})')

            env.close()

            # Compute candidate rating from gauntlet
            if matchup_results:
                candidate_rating = compute_elo_mle(matchup_results, ref_elos)
            else:
                candidate_rating = policy.rating

            # Check promotion criteria
            criteria = {}

            # Criterion 1: Rating didn't collapse (>-50 drop)
            # Both-sides play + small sample sizes create rating noise;
            # -50 catches real collapses while allowing statistical variance
            rating_delta = candidate_rating - policy.rating
            criteria['rating_no_collapse'] = {
                'passed': rating_delta > -50,
                'detail': f'delta={rating_delta:+.0f} (threshold: >-50)',
            }

            # Criterion 2: Beat own best checkpoint >= 40%
            # With 30 games, a truly equal policy has ~87% chance of hitting 40%
            # (vs only ~29% chance of hitting 55%). This catches real degradation
            # while accepting candidates at parity with gen0.
            own_best_key = f'own_best_gen{policy.best_checkpoint_generation}'
            if own_best_key in gauntlet:
                own_wr = gauntlet[own_best_key]['win_rate']
                criteria['beat_own_best'] = {
                    'passed': own_wr >= 0.40,
                    'detail': f'{own_wr:.0%} (threshold: >=40%)',
                }
            else:
                criteria['beat_own_best'] = {
                    'passed': True,
                    'detail': 'No own-best reference (auto-pass)',
                }

            # Criterion 3: No catastrophic regression (<30% where prev >=50%)
            worst_regression = None
            for ref in references:
                ref_id = ref['id']
                if ref_id == own_best_key:
                    continue  # Skip own-best for regression check

                current_wr = gauntlet[ref_id]['win_rate']

                # Find previous win rate from last round's win matrix
                prev_wr = self._get_previous_win_rate(policy_id, ref_id)

                if prev_wr is not None and prev_wr >= 0.50 and current_wr < 0.30:
                    if worst_regression is None or current_wr < worst_regression[1]:
                        worst_regression = (ref_id, current_wr, prev_wr)

            if worst_regression:
                criteria['no_regression'] = {
                    'passed': False,
                    'detail': (f'vs {worst_regression[0]}: {worst_regression[1]:.0%} '
                               f'(was {worst_regression[2]:.0%}, threshold: >=30%)'),
                }
            else:
                criteria['no_regression'] = {
                    'passed': True,
                    'detail': 'No regression below 30% on previously-beaten refs',
                }

            # Decision
            all_passed = all(c['passed'] for c in criteria.values())
            decision = 'PROMOTED' if all_passed else 'REJECTED'

            # Build rejection reason
            rejection_reason = None
            if not all_passed:
                failed = [f'{name}: {c["detail"]}'
                          for name, c in criteria.items() if not c['passed']]
                rejection_reason = '; '.join(failed)

            results[policy_id] = {
                'candidate_generation': policy.generation + 1,
                'gauntlet_results': gauntlet,
                'rating_before': policy.rating,
                'rating_after': candidate_rating,
                'rating_delta': rating_delta,
                'criteria': criteria,
                'decision': decision,
                'rejection_reason': rejection_reason,
            }

            symbol = '✓' if decision == 'PROMOTED' else '✗'
            print(f'[VERIFY] {symbol} {policy_id}: {decision}')
            if rejection_reason:
                print(f'         Reason: {rejection_reason}')

        return results

    def _get_previous_win_rate(self, policy_id, ref_id):
        """Get win rate of policy vs ref from the most recent round's win matrix."""
        if not self.manifest.rounds:
            return None

        last_round = self.manifest.rounds[-1]
        wm = last_round.win_matrix
        if not wm or 'labels' not in wm or 'data' not in wm:
            return None

        labels = wm['labels']
        if policy_id not in labels or ref_id not in labels:
            return None

        i = labels.index(policy_id)
        j = labels.index(ref_id)
        return wm['data'][i][j]

    def promote_or_reject(self, candidates, verification_results):
        """Apply verification decisions to manifest.

        Promoted: update model_path, best_checkpoint, generation++, rejections=0
        Rejected: discard candidate, rejections++, check 3-strike flag
        """
        for policy_id, result in verification_results.items():
            policy = self.manifest.get_policy_by_id(policy_id)
            if policy is None:
                continue

            decision = result['decision']
            round_num = self.manifest.next_round_number() - 1  # Current round

            if decision == 'PROMOTED':
                candidate_path = candidates.get(policy_id)
                if candidate_path is None:
                    continue

                # Move candidate to models dir as new generation
                next_gen = policy.generation + 1
                new_filename = f'{policy_id}_gen{next_gen}.pt'
                new_model_path = os.path.join(self.league_dir, 'models', new_filename)
                shutil.copy2(candidate_path, new_model_path)

                # Update policy entry
                policy.model_path = os.path.relpath(new_model_path, self.league_dir)
                policy.best_checkpoint_path = policy.model_path
                policy.best_checkpoint_generation = next_gen
                policy.generation = next_gen
                policy.rating = result.get('rating_after', policy.rating)
                policy.consecutive_rejections = 0
                policy.flagged_for_review = False

                self.manifest.update_rating_history(
                    policy_id, round_num, policy.rating, next_gen, 'PROMOTED')

                print(f'[LEAGUE] PROMOTED {policy_id} to gen {next_gen} '
                      f'(rating {policy.rating:.0f})')

            else:  # REJECTED
                policy.consecutive_rejections += 1
                if policy.consecutive_rejections >= 3:
                    policy.flagged_for_review = True
                    print(f'[LEAGUE] FLAGGED {policy_id} for review '
                          f'(3 consecutive rejections)')

                self.manifest.update_rating_history(
                    policy_id, round_num, policy.rating, policy.generation, 'REJECTED')

                print(f'[LEAGUE] REJECTED {policy_id} '
                      f'(rejections: {policy.consecutive_rejections}/3)')

        # Clean up candidate files
        candidate_dir = os.path.join(self.league_dir, 'candidates')
        if os.path.exists(candidate_dir):
            for f in os.listdir(candidate_dir):
                if f.endswith('_candidate.pt'):
                    os.remove(os.path.join(candidate_dir, f))

        self.manifest.save(self.manifest_path)

    def run_round(self, steps_per_policy=50_000_000, games_per_pair=30,
                  games_per_ref=100, num_envs=64, wandb_project=None,
                  num_workers=None):
        """Execute one full league round: eval → train → verify → promote/reject → re-eval."""
        round_num = self.manifest.next_round_number()
        print(f'\n{"="*60}')
        print(f'  LEAGUE ROUND {round_num}')
        print(f'{"="*60}\n')

        # Phase 1: Evaluate
        print(f'--- Phase 1: Evaluate ---')
        self.evaluate(games_per_pair, num_envs, num_workers)

        # Phase 2: Train
        print(f'\n--- Phase 2: Train ---')
        candidates = self.train_all(steps_per_policy, wandb_project)

        if not candidates:
            print('[LEAGUE] No candidates produced, skipping verify/promote')
            return

        # Phase 3: Verify
        print(f'\n--- Phase 3: Verify ---')
        results = self.verify(candidates, games_per_ref, num_envs)

        # Phase 4: Promote/Reject
        print(f'\n--- Phase 4: Promote/Reject ---')
        self.promote_or_reject(candidates, results)

        # Store verification in round record
        if self.manifest.rounds:
            last_round = self.manifest.rounds[-1]
            last_round.verification = results
            last_round.type = 'train_verify_eval'
            last_round.training_steps = steps_per_policy

        # Phase 5: Re-evaluate
        print(f'\n--- Phase 5: Re-evaluate ---')
        self.evaluate(games_per_pair, num_envs, num_workers)

        self.manifest.save(self.manifest_path)
        print(f'\n[LEAGUE] Round {round_num} complete!')

    def status(self):
        """Print league status summary."""
        active = self.manifest.get_active_policies()
        frozen = self.manifest.get_frozen_anchors()

        print(f'\n{"="*60}')
        print(f'  LEAGUE STATUS')
        print(f'{"="*60}')
        print(f'  Source: {self.manifest.source_project}')
        print(f'  Created: {self.manifest.created}')
        print(f'  Rounds completed: {len(self.manifest.rounds)}')
        print(f'  Active policies: {len(active)}')
        print(f'  Frozen anchors: {len(frozen)}')
        print(f'  Total policies: {len(self.manifest.policies)}')

        # Group by obs_scheme
        schemes = {}
        for p in self.manifest.policies:
            schemes.setdefault(p.obs_scheme, []).append(p)

        for scheme in sorted(schemes.keys()):
            policies = schemes[scheme]
            n_active = sum(1 for p in policies if p.status == 'active')
            n_frozen = sum(1 for p in policies if p.status == 'frozen')
            print(f'\n  obs_scheme {scheme}: {n_active} active, {n_frozen} frozen')

        # Top rated
        if active:
            print(f'\n  Top 5 Active Policies:')
            for p in sorted(active, key=lambda x: -x.rating)[:5]:
                flag = ' [FLAGGED]' if p.flagged_for_review else ''
                rej = f' (rej:{p.consecutive_rejections})' if p.consecutive_rejections > 0 else ''
                print(f'    {p.id[:45]}: '
                      f'rating={p.rating:.0f} gen={p.generation} '
                      f'scheme={p.obs_scheme}{rej}{flag}')

        # Recent round info
        if self.manifest.rounds:
            last = self.manifest.rounds[-1]
            print(f'\n  Last round: {last.round} ({last.type}) at {last.timestamp}')
            if last.verification:
                promoted = sum(1 for v in last.verification.values()
                               if v.get('decision') == 'PROMOTED')
                rejected = sum(1 for v in last.verification.values()
                               if v.get('decision') == 'REJECTED')
                print(f'  Verification: {promoted} promoted, {rejected} rejected')

        print()


def main():
    parser = argparse.ArgumentParser(description='Dogfight League System')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # collect
    collect_parser = subparsers.add_parser('collect', help='Collect policies from W&B')
    collect_parser.add_argument('--project', type=str, required=True)
    collect_parser.add_argument('--top-n', type=int, default=20)
    collect_parser.add_argument('--output-dir', type=str, default='league/')
    collect_parser.add_argument('--metric', type=str, default='environment/strength')
    collect_parser.add_argument('--entity', type=str, default=None)

    # eval
    eval_parser = subparsers.add_parser('eval', help='Run round-robin evaluation')
    eval_parser.add_argument('--manifest', type=str, required=True)
    eval_parser.add_argument('--games-per-pair', type=int, default=30)
    eval_parser.add_argument('--num-envs', type=int, default=64)
    eval_parser.add_argument('--num-workers', type=int, default=None,
                             help='Parallel worker processes (default: auto)')
    eval_parser.add_argument('--device', type=str, default='cuda')

    # train
    train_parser = subparsers.add_parser('train', help='Train all active policies')
    train_parser.add_argument('--manifest', type=str, required=True)
    train_parser.add_argument('--steps', type=int, default=50_000_000)
    train_parser.add_argument('--device', type=str, default='cuda')
    train_parser.add_argument('--wandb-project', type=str, default=None)

    # verify
    verify_parser = subparsers.add_parser('verify', help='Run verification gauntlet')
    verify_parser.add_argument('--manifest', type=str, required=True)
    verify_parser.add_argument('--games-per-ref', type=int, default=100)
    verify_parser.add_argument('--num-envs', type=int, default=64)
    verify_parser.add_argument('--device', type=str, default='cuda')

    # round
    round_parser = subparsers.add_parser('round', help='Run full league round')
    round_parser.add_argument('--manifest', type=str, required=True)
    round_parser.add_argument('--steps', type=int, default=50_000_000)
    round_parser.add_argument('--games-per-pair', type=int, default=30)
    round_parser.add_argument('--games-per-ref', type=int, default=100)
    round_parser.add_argument('--num-envs', type=int, default=64)
    round_parser.add_argument('--num-workers', type=int, default=None,
                              help='Parallel worker processes (default: auto)')
    round_parser.add_argument('--device', type=str, default='cuda')
    round_parser.add_argument('--wandb-project', type=str, default=None)

    # status
    status_parser = subparsers.add_parser('status', help='Show league status')
    status_parser.add_argument('--manifest', type=str, required=True)

    args = parser.parse_args()

    if args.command == 'collect':
        from pufferlib.ocean.dogfight.collect_from_wandb import collect_from_wandb
        collect_from_wandb(
            project=args.project,
            top_n=args.top_n,
            output_dir=args.output_dir,
            metric=args.metric,
            entity=args.entity,
        )

    elif args.command == 'eval':
        device = args.device
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        league = League(args.manifest, device)
        league.evaluate(args.games_per_pair, args.num_envs, args.num_workers)

    elif args.command == 'train':
        device = args.device
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        league = League(args.manifest, device)
        league.train_all(args.steps, args.wandb_project)

    elif args.command == 'verify':
        device = args.device
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        league = League(args.manifest, device)

        # Find candidates
        candidate_dir = os.path.join(league.league_dir, 'candidates')
        candidates = {}
        if os.path.exists(candidate_dir):
            for f in os.listdir(candidate_dir):
                if f.endswith('_candidate.pt'):
                    # Extract policy_id from filename: {policy_id}_gen{N}_candidate.pt
                    parts = f.rsplit('_gen', 1)
                    if len(parts) == 2:
                        policy_id = parts[0]
                        candidates[policy_id] = os.path.join(candidate_dir, f)

        if not candidates:
            print('[LEAGUE] No candidates found. Run train first.')
        else:
            results = league.verify(candidates, args.games_per_ref, args.num_envs)
            league.promote_or_reject(candidates, results)

    elif args.command == 'round':
        device = args.device
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        league = League(args.manifest, device)
        league.run_round(
            steps_per_policy=args.steps,
            games_per_pair=args.games_per_pair,
            games_per_ref=args.games_per_ref,
            num_envs=args.num_envs,
            wandb_project=args.wandb_project,
            num_workers=args.num_workers,
        )

    elif args.command == 'status':
        league = League(args.manifest, 'cpu')
        league.status()

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
