// Compile all SynthDef sources in audio/synthdefs/ into web/synthdefs/*.scsyndef.
// Run headless with:  sclang audio/compile_synthdefs.scd
(
var here = thisProcess.nowExecutingPath.dirname;
var outDir = here.dirname +/+ "web" +/+ "synthdefs";
var sources = ["grain_cloud_buf.scd", "grain_cloud_sin.scd", "sub_drone.scd"];

File.mkdir(outDir);

sources.do { |file|
	var def = (here +/+ "synthdefs" +/+ file).load;
	def.writeDefFile(outDir);
	("wrote" + (outDir +/+ def.name ++ ".scsyndef")).postln;
};

0.exit;
)
