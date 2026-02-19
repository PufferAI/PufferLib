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
import traceback
from datetime import datetime

import torch

from pufferlib.ocean.dogfight.dogfight_log import init_log, log


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
        log(f'[EVAL] policies={len(all_policies)} games_per_pair={games_per_pair} num_envs={num_envs}')

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
        log(f'[EVAL] event=complete round={round_num}')
        return win_matrix, ratings

    def train_all(self, steps_per_policy=50_000_000, wandb_project=None):
        """Train ALL active policies sequentially.

        Returns dict: policy_id -> candidate_checkpoint_path
        """
        from pufferlib.ocean.dogfight.train_dual_selfplay import train_league_round

        active = self.manifest.get_active_policies()
        candidates = {}

        log(f'[TRAIN] event=all_start policies={len(active)} steps={steps_per_policy}')

        for i, policy in enumerate(active):
            if policy.flagged_for_review:
                policy.flagged_for_review = False
                policy.consecutive_rejections = 0
                self.manifest.save(self.manifest_path)
                log(f'[TRAIN] policy={policy.id} event=unflagged reason=auto_reset')

            # Check for existing candidate (crash recovery)
            candidate_dir = os.path.join(self.league_dir, 'candidates')
            next_gen = policy.generation + 1
            candidate_filename = f'{policy.id}_gen{next_gen}_candidate.pt'
            candidate_path = os.path.join(candidate_dir, candidate_filename)

            if os.path.exists(candidate_path):
                log(f'[TRAIN] policy={policy.id} event=skipped reason=candidate_exists')
                candidates[policy.id] = candidate_path
                continue

            log(f'[TRAIN] policy={policy.id} event=start index={i+1}/{len(active)}')
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
                    log(f'[TRAIN] policy={policy.id} event=done elapsed={elapsed:.0f}s')
            except Exception as e:
                log(f'[ERROR] policy={policy.id} phase=train msg="{e}"')
                log(f'[ERROR] policy={policy.id} traceback={traceback.format_exc()}')

            # Save manifest after each policy (crash recovery)
            self.manifest.save(self.manifest_path)

        log(f'[TRAIN] event=all_done candidates={len(candidates)} total={len(active)}')
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

            log(f'[VERIFY] policy={policy_id} gen={policy.generation + 1} event=start')

            # Build reference set: all frozen anchors in same scheme
            references = []
            own_anchor = None

            same_scheme = self.manifest.get_policies_by_scheme(policy.obs_scheme)
            total_frozen = 0
            skipped_missing = 0
            for ref in same_scheme:
                if ref.status == 'frozen':
                    total_frozen += 1
                    ref_path = os.path.join(self.league_dir, ref.model_path)
                    if os.path.exists(ref_path):
                        references.append({
                            'id': ref.id,
                            'path': ref_path,
                            'hidden_size': ref.hidden_size,
                        })
                        # Identify this policy's own gen-0 anchor
                        if ref.source_policy == policy_id:
                            own_anchor = ref
                    else:
                        skipped_missing += 1
                        log(f'[VERIFY] policy={policy_id} warning=missing_ref ref={ref.id} path={ref_path}')

            if skipped_missing > 0:
                log(f'[VERIFY] policy={policy_id} missing_refs={skipped_missing}/{total_frozen}')
                if skipped_missing > total_frozen / 2:
                    log(f'[ERROR] policy={policy_id} phase=verify msg=">50% references missing, skipping"')
                    results[policy_id] = {
                        'decision': 'REJECTED',
                        'reason': f'{skipped_missing}/{total_frozen} reference models missing',
                        'gauntlet_results': {},
                    }
                    continue

            if not references:
                log(f'[VERIFY] policy={policy_id} event=auto_promote reason=no_references')
                results[policy_id] = {
                    'decision': 'PROMOTED',
                    'reason': 'No references available',
                    'gauntlet_results': {},
                }
                continue

            # Create env for this scheme (eval_spawn_mode=2 for symmetric fair spawns)
            # report_interval=999999 prevents vec_log from being called during
            # env.step(), so clean_fights accumulate in the C log struct across
            # all gauntlet matches. We read them once after the gauntlet via vec_log.
            env = Dogfight(
                num_envs=1,
                render_mode=None,
                obs_scheme=policy.obs_scheme,
                curriculum_enabled=1,
                curriculum_randomize=1,
                eval_spawn_mode=2,
                fixed_stage=20,
                max_steps=6000,
                report_interval=999999,
            )
            binding.vec_enable_opponent_override(env.c_envs, 1)
            binding.vec_set_selfplay_active(env.c_envs, 1)

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
                                 half, self.device,
                                 player_hidden_size=policy.hidden_size,
                                 opponent_hidden_size=ref['hidden_size'])
                r2 = run_matches(env, ref_policy, candidate_policy,
                                 games_per_ref - half, self.device,
                                 player_hidden_size=ref['hidden_size'],
                                 opponent_hidden_size=policy.hidden_size)
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

                log(f'[VERIFY] policy={policy_id} vs={ref["id"]} w={result["wins"]} l={result["losses"]} d={result["draws"]} wr={win_rate*100:.1f}')

            # Read accumulated clean fight stats before closing env.
            # vec_log returns clean_fights already averaged by n (= clean_fights_count / n).
            log_data = binding.vec_log(env.c_envs)
            clean_fight_rate = log_data.get('clean_fights', 0.0) if log_data else 0.0
            player_ground = log_data.get('player_ground', 0.0) if log_data else 0.0
            opp_ground = log_data.get('opp_ground', 0.0) if log_data else 0.0
            log(f'[VERIFY] policy={policy_id} clean_fight_rate={clean_fight_rate:.1%} player_ground={player_ground:.1%} opp_ground={opp_ground:.1%}')

            env.close()

            # Compute candidate rating from gauntlet
            if matchup_results:
                candidate_rating = compute_elo_mle(matchup_results, ref_elos)
            else:
                candidate_rating = policy.rating

            # Check promotion criteria
            criteria = {}

            # Criterion 1: Rating didn't collapse vs own anchor
            # Uses anchor rating (fixed reference) instead of volatile round-robin rating
            if own_anchor is not None:
                anchor_rating = own_anchor.rating
                rating_delta = candidate_rating - anchor_rating
                criteria['rating_no_collapse'] = {
                    'passed': rating_delta > -100,
                    'detail': f'delta={rating_delta:+.0f} vs anchor {own_anchor.id[:20]} '
                              f'(anchor={anchor_rating:.0f}, threshold: >-100)',
                }
            else:
                # No anchor — fall back to policy's current rating
                rating_delta = candidate_rating - policy.rating
                criteria['rating_no_collapse'] = {
                    'passed': rating_delta > -100,
                    'detail': f'delta={rating_delta:+.0f} vs policy rating '
                              f'(no anchor found, threshold: >-100)',
                }

            # Criterion 2: Beat own anchor >= 55%
            # Must demonstrably improve over previous generation, not just break even.
            if own_anchor is not None and own_anchor.id in gauntlet:
                own_anchor_wr = gauntlet[own_anchor.id]['win_rate']
                criteria['beat_own_anchor'] = {
                    'passed': own_anchor_wr >= 0.55,
                    'detail': f'{own_anchor_wr:.0%} vs {own_anchor.id[:20]} (threshold: >=55%)',
                }
            else:
                criteria['beat_own_anchor'] = {
                    'passed': True,
                    'detail': 'No own anchor found (auto-pass)',
                }

            # Criterion 3: No anchor regression — beat ALL anchors >= 35%
            # Catches cases where training improved vs one anchor but collapsed vs another
            worst_anchor_wr = None
            worst_anchor_id = None
            for ref in references:
                ref_id = ref['id']
                ref_policy = self.manifest.get_policy_by_id(ref_id)
                if ref_policy is None or ref_policy.status != 'frozen':
                    continue
                if ref_id in gauntlet:
                    wr = gauntlet[ref_id]['win_rate']
                    if worst_anchor_wr is None or wr < worst_anchor_wr:
                        worst_anchor_wr = wr
                        worst_anchor_id = ref_id

            if worst_anchor_wr is not None and worst_anchor_wr < 0.35:
                criteria['all_anchors'] = {
                    'passed': False,
                    'detail': f'vs {worst_anchor_id[:20]}: {worst_anchor_wr:.0%} (threshold: >=35%)',
                }
            else:
                detail = 'All anchors >= 35%'
                if worst_anchor_wr is not None:
                    detail = f'Worst: {worst_anchor_wr:.0%} vs {worst_anchor_id[:20]} (threshold: >=35%)'
                criteria['all_anchors'] = {
                    'passed': True,
                    'detail': detail,
                }

            # Criterion 4: No catastrophic regression (<30% where prev >=50%)
            worst_regression = None
            for ref in references:
                ref_id = ref['id']
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

            # Criterion 5: Clean fight rate >= 80%
            # Neither plane should be crashing into the ground. Clean = kill or timeout, not crash.
            criteria['clean_fights'] = {
                'passed': clean_fight_rate >= 0.80,
                'detail': f'{clean_fight_rate:.0%} clean fights (threshold: >=80%)',
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

            log(f'[VERDICT] policy={policy_id} result={decision} reason="{rejection_reason or "all_criteria_passed"}"')

        return results

    def _get_previous_win_rate(self, policy_id, ref_id):
        """Get win rate of policy vs ref from the second-to-last round's win matrix.

        During run_round(), Phase 1 (evaluate) appends the current eval round.
        The regression check needs the PREVIOUS round's data (before this eval),
        so we use rounds[-2] when available, falling back to rounds[-1] if only
        one round exists.
        """
        if not self.manifest.rounds:
            return None

        # Use second-to-last round if available (skip current eval round)
        idx = -2 if len(self.manifest.rounds) >= 2 else -1
        prev_round = self.manifest.rounds[idx]
        wm = prev_round.win_matrix
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

                log(f'[PROMOTE] policy={policy_id} gen={policy.generation-1}->{next_gen} rating={policy.rating:.0f}')

            else:  # REJECTED
                policy.consecutive_rejections += 1
                if policy.consecutive_rejections >= 3:
                    policy.flagged_for_review = True
                    log(f'[PROMOTE] policy={policy_id} result=flagged rejections={policy.consecutive_rejections}/3')

                    # Reset to anchor weights for fresh start next round
                    same_scheme = self.manifest.get_policies_by_scheme(policy.obs_scheme)
                    own_anchor = next((r for r in same_scheme
                                       if r.status == 'frozen' and r.source_policy == policy_id), None)
                    if own_anchor is not None:
                        anchor_path = os.path.join(self.league_dir, own_anchor.model_path)
                        policy_path = os.path.join(self.league_dir, policy.model_path)
                        if os.path.exists(anchor_path):
                            shutil.copy2(anchor_path, policy_path)
                            log(f'[PROMOTE] policy={policy_id} event=anchor_reset anchor={own_anchor.id}')

                self.manifest.update_rating_history(
                    policy_id, round_num, policy.rating, policy.generation, 'REJECTED')

                log(f'[PROMOTE] policy={policy_id} result=rejected rejections={policy.consecutive_rejections}/3')

        # Clean up only candidate files that were processed in this round
        for policy_id in verification_results:
            candidate_path = candidates.get(policy_id)
            if candidate_path and os.path.exists(candidate_path):
                os.remove(candidate_path)

        self.manifest.save(self.manifest_path)

    def run_round(self, steps_per_policy=50_000_000, games_per_pair=30,
                  games_per_ref=100, num_envs=64, wandb_project=None,
                  num_workers=None):
        """Execute one full league round: eval → train → verify → promote/reject → re-eval."""
        round_num = self.manifest.next_round_number()
        log(f'[ROUND] num={round_num} event=start')

        # Phase 1: Evaluate
        log(f'[PHASE] round={round_num} phase=eval')
        self.evaluate(games_per_pair, num_envs, num_workers)

        # Phase 2: Train
        log(f'[PHASE] round={round_num} phase=train')
        candidates = self.train_all(steps_per_policy, wandb_project)

        if not candidates:
            log(f'[TRAIN] event=no_candidates round={round_num}')
            return

        # Phase 3: Verify
        log(f'[PHASE] round={round_num} phase=verify')
        results = self.verify(candidates, games_per_ref, num_envs)

        # Phase 4: Promote/Reject
        log(f'[PHASE] round={round_num} phase=promote')
        self.promote_or_reject(candidates, results)

        # Store verification as a separate round record
        from pufferlib.ocean.dogfight.league_manifest import RoundRecord
        verify_round = RoundRecord(
            round=self.manifest.next_round_number(),
            type='train_verify',
            timestamp=datetime.now().isoformat(),
            win_matrix={'labels': [], 'data': []},
            ratings={},
            verification=results,
            training_steps=steps_per_policy,
        )
        self.manifest.add_round(verify_round)

        # Phase 5: Re-evaluate
        log(f'[PHASE] round={round_num} phase=reeval')
        self.evaluate(games_per_pair, num_envs, num_workers)

        self.manifest.save(self.manifest_path)
        log(f'[ROUND] num={round_num} event=complete')

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

    if args.command in ('eval', 'train', 'verify', 'round'):
        init_log('league/logs', f'league_{args.command}')

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
