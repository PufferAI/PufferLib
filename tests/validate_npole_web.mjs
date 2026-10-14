// Browser qualification of the built live demo. Requires Playwright.
// node tests/validate_npole_web.mjs URL NEW_OUTPUT_DIR FIRST_SEED EPISODES ASSETS_DIR
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import zlib from 'node:zlib';

const [url, out, firstText, countText, assets] = process.argv.slice(2);
const first = Number(firstText);
const count = Number(countText);
if (!url || !out || !assets || !Number.isInteger(first) || !Number.isInteger(count)
    || first < 0 || count < 1 || first + count > 2 ** 32 || fs.existsSync(out)) {
    throw new Error('Provide URL, new output directory, uint32 first seed, episode count and built assets directory');
}
fs.mkdirSync(out, { recursive: true });
const sha = buffer => crypto.createHash('sha256').update(buffer).digest('hex');
const buildHashes = Object.fromEntries(['game.html','game.js','game.wasm','game.data'].map(name =>
    [name, sha(fs.readFileSync(path.join(assets, name)))]));
const frozen = { frozen_at: new Date().toISOString(), engine: 'browser_wasm_rk4',
    seeds: Array.from({length:count}, (_,i)=>first+i), control_hz:50, physics_hz:1000,
    delay_ms:20, build_sha256:buildHashes, training_during_evaluation:false };
fs.writeFileSync(path.join(out, 'frozen.json'), JSON.stringify(frozen, null, 2));
const options = {headless:true, args:['--no-sandbox','--use-angle=swiftshader','--enable-unsafe-swiftshader']};
if (process.env.NPOLE_CHROMIUM_PATH) options.executablePath = process.env.NPOLE_CHROMIUM_PATH;
const browser = await chromium.launch(options);
const page = await browser.newPage({viewport:{width:1050,height:1100}});
const errors = [];
page.on('pageerror', error => errors.push(String(error)));
page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
try {
    await page.goto(url);
    await page.waitForFunction(() => window.npoleReady === true, null, {timeout:30000});
    const rows = [];
    for (const seed of frozen.seeds) {
        const started = performance.now();
        const trace = await page.evaluate(seed => {
            const call = (name, type='number', types=[], args=[]) => Module.ccall(name,type,types,args);
            const state = () => Array.from({length:22},(_,i)=>call('np_demo_state','number',['number'],[i]));
            call('np_demo_reset',null,['number'],[seed]);
            const states=[state()], commands=[], applied=[], pending=[], inferred=[], residuals=[];
            for (let tick=0;tick<1000;tick++) {
                if (call('np_demo_advance','number',['number'],[1]) !== tick+1) break;
                states.push(state());
                commands.push(call('np_demo_command'));
                applied.push(call('np_demo_applied'));
                pending.push(call('np_demo_pending'));
                residuals.push(call('np_demo_residual'));
                inferred.push(Array.from({length:22},(_,i)=>call('np_demo_estimate','number',['number'],[i])));
                if (call('np_demo_failed')) break;
            }
            call('np_demo_toggle_pause',null);
            return {seed, states, commands, applied, pending, inferred, residuals,
                reported_success:!!call('np_demo_success'), reported_failed:!!call('np_demo_failed'),
                hold_ticks:call('np_demo_hold')};
        }, seed);
        const good = state => state.every(Number.isFinite) && Math.abs(state[0]) < 2.4
            && state.slice(1,11).every(x => Math.abs(Math.atan2(Math.sin(x),Math.cos(x))) < .22)
            && state.slice(12).every(x => Math.abs(x) < 1);
        const queue = trace.applied.every((value,i)=>value === (i ? trace.commands[i-1] : 0))
            && trace.pending.every((value,i)=>value === trace.commands[i]);
        const bounded = trace.commands.every(x=>Number.isFinite(x) && Math.abs(x)<=1)
            && trace.states.every(state=>state.every(Number.isFinite) && Math.abs(state[0])<2.4);
        const endpoints = trace.states.length === 1001 && trace.states.slice(-251).every(good);
        const success = queue && bounded && endpoints && trace.reported_success && !trace.reported_failed;
        const recording = `wasm-${seed}.json.gz`;
        const compressed = zlib.gzipSync(JSON.stringify(trace));
        fs.writeFileSync(path.join(out, recording), compressed);
        const row = {seed, success, ticks:trace.commands.length, hold_ticks:trace.hold_ticks,
            all_251_final_endpoints_good:endpoints, queue_exact:queue, force_and_rail_bounds:bounded,
            max_inverse_residual:Math.max(0,...trace.residuals),
            maximum_decode_error:Math.max(0,...trace.inferred.map((state,t)=>
                Math.max(...state.map((value,i)=>Math.abs(value-trace.states[t][i]))))),
            wall_seconds:(performance.now()-started)/1000, recording, replay_sha256:sha(compressed)};
        rows.push(row);
        console.log(JSON.stringify(row));
    }
    for (const [name,digest] of Object.entries(buildHashes)) {
        if (sha(fs.readFileSync(path.join(assets,name))) !== digest) throw new Error('Build changed during evaluation');
    }
    const receipt = {schema:'rage_mill.puffer_wasm_validation.v1', configuration:frozen,
        browser:browser.version(), trials:rows, successes:rows.filter(row=>row.success).length, errors};
    fs.writeFileSync(path.join(out,'receipt.json'),JSON.stringify(receipt,null,2));
    if (errors.length || rows.some(row=>!row.success)) process.exitCode=2;
} finally {
    await browser.close();
}
