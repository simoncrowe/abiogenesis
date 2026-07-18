// sub-drone: one near-static low tone anchoring the low end.
//
// The only survivor of the "drone" concept from doc 14, reduced from a chord
// progression to a single slowly re-pitched sine/triangle. The conductor moves
// `freq` rarely (on some scene changes) by a ratio from the pitch set.

SynthDef('sub-drone', {
	|freq = 55, amp = 0.2, attack = 8, release = 8, gate = 1, out_bus = 0|

	var freqC, ampC, sig, env, move;

	freqC = freq.clip(20, 200);
	ampC  = amp.clip(0, 1);

	// Sine + a touch of detuned triangle + sub-octave for weight.
	sig = SinOsc.ar(freqC)
		+ (LFTri.ar(freqC * 1.001) * 0.3)
		+ (SinOsc.ar(freqC * 0.5) * 0.4);
	sig = sig * (1 / 1.7);

	// Very slow amplitude breathing so it isn't perfectly static.
	move = 1 + SinOsc.kr(0.06, 0, 0.06);
	sig = sig * move;

	env = EnvGen.kr(Env.asr(attack.max(0), 1, release.max(0.01)), gate, doneAction: 2);

	sig = Pan2.ar(sig, 0, ampC * env);

	Out.ar(out_bus, sig);
}).writeDefFile(~outDir);
