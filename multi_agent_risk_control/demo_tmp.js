
(() => {
  const canvas = document.getElementById("cv");
  const ctx = canvas.getContext("2d");

  // ---------- Config (meters, seconds) -----------
  const cfg = {
    arena: 34.0,        // m
    dtPhys: 0.05,       // physics tick (s)
    N: 7,
    M: 18,

    // safety
    safePed: 2.0,       // m (keep-out ring)
    safeDrone: 1.5,     // m (soft keep-out)

    // collision thresholds (reset triggers)
    collPed: 0.55,      // m
    collDrone: 0.80,    // m

    // drone dynamics
    vMax: 4.4,          // m/s
    vPref: 3.2,         // m/s
    aMax: 3.2,          // m/s^2
    tau: 0.60,          // s

    // planner
    H: 18,
    K: 7,
    coneDeg: 65,
    speedLevels: [0.75, 1.0, 1.15],

    // cost weights
    kp: 3.0,
    kr: 2.2,
    lamMin: 0.22,
    goalW: 0.22,
    riskW: 1.0,
    turnW: 0.15,

    // CDT updates
    epsP: 0.10,
    epsR: 0.08,
    etaP: 0.055,
    etaR: 0.055,
    lamLo: 0.22,
    lamHi: 4.0,
    lamInit: 3.0,

    // rates
    planHz: 18,

    // visuals
    pedDirLen: 1.4,
    samplesPerSeg: 7,
  };

  // ---------- RNG ----------
  const rng = (seed => {
    let s = seed >>> 0;
    return () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296;
  })(2026);

  function rand(a,b){ return a + (b-a)*rng(); }
  function randn(){
    const u = Math.max(1e-9, rng());
    const v = Math.max(1e-9, rng());
    return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
  }

  // ---------- Vector helpers ----------
  const vadd = (a,b)=>[a[0]+b[0], a[1]+b[1]];
  const vsub = (a,b)=>[a[0]-b[0], a[1]-b[1]];
  const vmul = (a,s)=>[a[0]*s, a[1]*s];
  const vdot = (a,b)=>a[0]*b[0]+a[1]*b[1];
  const vnorm = (a)=>Math.hypot(a[0],a[1]);
  const vunit = (a)=>{
    const n=vnorm(a);
    return n<1e-9 ? [0,0] : [a[0]/n, a[1]/n];
  };
  const clipNorm = (a, maxn)=>{
    const n=vnorm(a);
    if(n<=maxn) return a;
    const s=maxn/(n+1e-9);
    return [a[0]*s, a[1]*s];
  };
  const relu = (x)=>Math.max(0,x);

  // ---------- Spline ----------
  function catmullRom(points, samplesPerSeg){
    if(points.length < 4) return points.slice();
    const p = [points[0], ...points, points[points.length-1]];
    const out = [];
    for(let i=0;i<points.length-1;i++){
      const P0=p[i], P1=p[i+1], P2=p[i+2], P3=p[i+3];
      for(let j=0;j<samplesPerSeg;j++){
        const t=j/samplesPerSeg, t2=t*t, t3=t2*t;
        const x=0.5*((2*P1[0])+(-P0[0]+P2[0])*t+(2*P0[0]-5*P1[0]+4*P2[0]-P3[0])*t2+(-P0[0]+3*P1[0]-3*P2[0]+P3[0])*t3);
        const y=0.5*((2*P1[1])+(-P0[1]+P2[1])*t+(2*P0[1]-5*P1[1]+4*P2[1]-P3[1])*t2+(-P0[1]+3*P1[1]-3*P2[1]+P3[1])*t3);
        out.push([x,y]);
      }
    }
    out.push(points[points.length-1]);
    return out;
  }

  // ---------- State ----------
  let pedPos, pedVel;
  let tSim = 0;
  let paused = false;

  // overlay flags
  let overlaysOn = true;

  const COLORS = {
    blue:  "#1f77b4",
    purple:"#9467bd",
    orange:"#ff7f0e",
    red:   "#d62728",
  };

  // ---------- Channels (paper mapping) ----------
  // Channels c ∈ C are:
  //   - ped(i): drone i separation-to-pedestrians
  //   - pair(i,k): inter-drone separation for unordered pair (i<k)
  // Losses are ℓ^c_t ∈ [0,1] and m_t = max_c ℓ^c_t.
  // Global update (Algorithm 1): λ^G_{t+1} = Π[λlo,λhi](λ^G_t + η(ε - m_t)).
  // Effective parameters: λ^{e,c}_t = λ^G_t if c ∈ A_t else local λ^c_t.
  // Active set A_t is chosen from past information (we use the previous maximizer),
  // and we also always apply λ^G to the *current* worst channel measured from the current state.

  function makeWorld(name, color){
    return {
      name, color,
      pos: [], vel: [], goal: [],
      // planning output
      vdes: [],
      plan: [],
      // reward / goals
      reward: 0,
      // collisions
      crosses: [], // {x,y,until}
      // series
      dSeries: [],
      rSeries: [],
      cSeries: [], // collision markers for distance plot: {t, d}
      // controller parameters
      lamG: cfg.lamInit,     // global λ^G
      lamPed: null,          // local λ^{ped(i)}
      lamPair: null,         // local λ^{pair(i,k)}
      prevMaxChan: null,     // previous maximizer channel (for A_t)
      // metrics
      cumColl: 0,
      avgLam: cfg.lamInit,
    };
  }

  let W0, W1, W2;

  function initPedestrians(){
    pedPos = new Array(cfg.M);
    pedVel = new Array(cfg.M);
    for(let j=0;j<cfg.M;j++){
      pedPos[j] = [rand(7, cfg.arena-7), rand(5, cfg.arena-5)];
      const ang = rand(0, Math.PI*2);
      const sp = rand(0.9, 1.7);
      pedVel[j] = [Math.cos(ang)*sp, Math.sin(ang)*sp];
    }
  }

  function randomGoalOnSide(side){
    const x = side==="L" ? rand(2.0, 5.5) : rand(cfg.arena-5.5, cfg.arena-2.0);
    const y = rand(5.0, cfg.arena-5.0);
    return [x,y];
  }

  function safeSpawnOnSide(side){
    for(let tries=0; tries<60; tries++){
      const x = side==="L" ? rand(2.0, 5.5) : rand(cfg.arena-5.5, cfg.arena-2.0);
      const y = rand(5.0, cfg.arena-5.0);
      const p = [x,y];
      let ok = true;
      for(let j=0;j<cfg.M;j++){
        if(vnorm(vsub(p, pedPos[j])) < cfg.safePed*0.9){ ok=false; break; }
      }
      if(ok) return p;
    }
    return side==="L" ? [3.5, rand(5.0, cfg.arena-5.0)] : [cfg.arena-3.5, rand(5.0, cfg.arena-5.0)];
  }

  function initDrones(world){
    world.pos = new Array(cfg.N);
    world.vel = new Array(cfg.N);
    world.goal = new Array(cfg.N);
    world.vdes = new Array(cfg.N);
    world.plan = new Array(cfg.N);

    for(let i=0;i<cfg.N;i++){
      world.pos[i] = [rand(2.0, 5.5), rand(5.0, cfg.arena-5.0)];
      world.vel[i] = [randn()*0.20, randn()*0.20];
      world.goal[i] = randomGoalOnSide("R");
      world.vdes[i] = [0,0];
      world.plan[i] = new Array(cfg.H+1).fill(0).map(()=>[0,0]);
    }
  }

  function initWorlds(){
    tSim = 0;
    initPedestrians();

    W0 = makeWorld("No risk control", COLORS.blue);
    W1 = makeWorld("Global risk control", COLORS.purple);
    W2 = makeWorld("Selective worst‑channel control", COLORS.orange);

    initDrones(W0);
    initDrones(W1);
    initDrones(W2);

    // lambda init (paper-consistent)
    // Global: one scalar λ^G applied to all channels.
    W1.lamG = cfg.lamInit;
    // Selective: global λ^G + locals per channel.
    W2.lamG = cfg.lamInit;
    W2.lamPed = new Array(cfg.N).fill(cfg.lamHi); // locals start aggressive
    W2.lamPair = Array.from({length: cfg.N}, () => new Array(cfg.N).fill(cfg.lamHi));
    for(let i=0;i<cfg.N;i++) W2.lamPair[i][i] = cfg.lamHi;

    // clear series
    for(const w of [W0,W1,W2]){
      w.reward = 0;
      w.cumColl = 0;
      w.crosses = [];
      w.dSeries = [];
      w.rSeries = [];
      w.cSeries = [];
    }
  }

  // ---------- Pedestrian dynamics ----------
  function stepPeds(){
    const low=1.3, high=cfg.arena-1.3;
    for(let j=0;j<cfg.M;j++){
      pedPos[j][0] += cfg.dtPhys * pedVel[j][0];
      pedPos[j][1] += cfg.dtPhys * pedVel[j][1];
      for(let d=0; d<2; d++){
        if(pedPos[j][d] < low){ pedPos[j][d]=low; pedVel[j][d]*=-1; }
        else if(pedPos[j][d] > high){ pedPos[j][d]=high; pedVel[j][d]*=-1; }
      }
    }
  }

  // ---------- Drone inertial step ----------
  function inertialStep(world){
    for(let i=0;i<cfg.N;i++){
      const v = world.vel[i];
      const vdes = world.vdes[i];
      let a = vmul(vsub(vdes, v), 1.0 / cfg.tau);
      a = clipNorm(a, cfg.aMax);
      let v2 = clipNorm(vadd(v, vmul(a, cfg.dtPhys)), cfg.vMax);
      let p2 = vadd(world.pos[i], vmul(v2, cfg.dtPhys));
      p2[0] = Math.min(cfg.arena-1.0, Math.max(1.0, p2[0]));
      p2[1] = Math.min(cfg.arena-1.0, Math.max(1.0, p2[1]));
      world.vel[i] = v2;
      world.pos[i] = p2;
    }
  }

  // ---------- Rewards / goal switching ----------
  function updateGoalsAndReward(world){
    for(let i=0;i<cfg.N;i++){
      const d = vnorm(vsub(world.goal[i], world.pos[i]));
      if(d < 1.1){
        world.reward += 1;
        const side = (world.goal[i][0] > cfg.arena/2) ? "L" : "R"; // flip
        world.goal[i] = randomGoalOnSide(side);
      }
    }
  }

  // ---------- Closest distance ----------
  function closestDistance(world){
    let dmin = Infinity;
    for(let i=0;i<cfg.N;i++){
      for(let j=0;j<cfg.M;j++){
        const d = vnorm(vsub(world.pos[i], pedPos[j]));
        if(d < dmin) dmin = d;
      }
    }
    for(let i=0;i<cfg.N;i++){
      for(let k=i+1;k<cfg.N;k++){
        const d = vnorm(vsub(world.pos[i], world.pos[k]));
        if(d < dmin) dmin = d;
      }
    }
    return dmin;
  }

  // ---------- Collision detection + reset ----------
  function addCross(world, x, y){
    world.crosses.push({x, y, until: tSim + 1.0}); // 1s visibility
  }

  function resetDrone(world, idx){
    // Respawn on the opposite side of the reward (goal) the drone is currently going to.
    const goalSide = (world.goal[idx][0] > cfg.arena/2) ? "R" : "L";
    const side = (goalSide === "R") ? "L" : "R";
    world.pos[idx] = safeSpawnOnSide(side);
    world.vel[idx] = [0,0];
  }

  function handleCollisions(world){
    // drone-ped collisions: reset drone
    for(let i=0;i<cfg.N;i++){
      for(let j=0;j<cfg.M;j++){
        const d = vnorm(vsub(world.pos[i], pedPos[j]));
        if(d < cfg.collPed){
          addCross(world, world.pos[i][0], world.pos[i][1]);
          world.cumColl += 1; // 1 collision event
          world.cSeries.push({t: tSim, d: closestDistance(world)});
          resetDrone(world, i);
          break;
        }
      }
    }
    // drone-drone collisions: reset both, count as one event
    for(let i=0;i<cfg.N;i++){
      for(let k=i+1;k<cfg.N;k++){
        const d = vnorm(vsub(world.pos[i], world.pos[k]));
        if(d < cfg.collDrone){
          const mid = vmul(vadd(world.pos[i], world.pos[k]), 0.5);
          addCross(world, mid[0], mid[1]);
          world.cumColl += 1; // 1 collision event
          world.cSeries.push({t: tSim, d: d});
          resetDrone(world, i);
          resetDrone(world, k);
        }
      }
    }
    world.crosses = world.crosses.filter(c => c.until > tSim);
  }

  // ---------- Planner prediction ----------
  function predictLinear(pos, vel, h){
    return [pos[0] + (h*cfg.dtPhys)*vel[0], pos[1] + (h*cfg.dtPhys)*vel[1]];
  }

  // ---------- Per-channel losses + maximizer (paper: m_t and argmax_c ℓ^c_t) ----------
  function computeLossesAndMax(world){
    const lossPed = new Array(cfg.N).fill(0.0);
    for(let i=0;i<cfg.N;i++){
      let dmin = Infinity;
      for(let j=0;j<cfg.M;j++){
        const d = vnorm(vsub(world.pos[i], pedPos[j]));
        if(d < dmin) dmin = d;
      }
      lossPed[i] = Math.min(1, Math.max(0, (cfg.safePed - dmin) / cfg.safePed));
    }
    const lossPair = Array.from({length: cfg.N}, ()=> new Array(cfg.N).fill(0.0));
    for(let i=0;i<cfg.N;i++){
      for(let k=i+1;k<cfg.N;k++){
        const d = vnorm(vsub(world.pos[i], world.pos[k]));
        const l = Math.min(1, Math.max(0, (cfg.safeDrone - d) / cfg.safeDrone));
        lossPair[i][k] = l;
        lossPair[k][i] = l;
      }
    }
    let mt = 0.0;
    let arg = {type:"ped", i:0};
    for(let i=0;i<cfg.N;i++){
      if(lossPed[i] > mt){ mt = lossPed[i]; arg = {type:"ped", i}; }
    }
    for(let i=0;i<cfg.N;i++){
      for(let k=i+1;k<cfg.N;k++){
        const l = lossPair[i][k];
        if(l > mt){ mt = l; arg = {type:"pair", i, k}; }
      }
    }
    return {lossPed, lossPair, mt, arg};
  }

  function planOne(world, i, controller, effLamPed, effLamPair){
    const p0 = world.pos[i], v0 = world.vel[i], g = world.goal[i];

    const baseDir = vunit(vsub(g, p0));
    const baseAng = Math.atan2(baseDir[1], baseDir[0]);
    const cone = (cfg.coneDeg * Math.PI / 180);
    const K = cfg.K;

    const angles = new Array(K);
    for(let k=0;k<K;k++){
      const u = (K===1)? 0.5 : (k/(K-1));
      angles[k] = baseAng + (u*2 - 1)*cone;
    }
    const speeds = cfg.speedLevels.map(s => s*cfg.vPref);

    // No-risk controller still includes penalties, but uses a fixed aggressive λ.
    const wP0 = cfg.kp / Math.max(effLamPed(i), cfg.lamMin);

    const other = [];
    for(let k=0;k<cfg.N;k++) if(k!==i) other.push(k);

    let bestCost = Infinity;
    let bestV = [0,0];
    let bestPath = null;

    for(let ak=0; ak<angles.length; ak++){
      const ang = angles[ak];
      for(let si=0; si<speeds.length; si++){
        const sp = speeds[si];
        const vdes = [Math.cos(ang)*sp, Math.sin(ang)*sp];

        let p = [p0[0], p0[1]];
        let v = [v0[0], v0[1]];
        const path = new Array(cfg.H+1);
        path[0] = [p[0], p[1]];

        let cost = 0.0;
        for(let h=1; h<=cfg.H; h++){
          // inertial rollout (single drone)
          let a = vmul(vsub(vdes, v), 1.0 / cfg.tau);
          a = clipNorm(a, cfg.aMax);
          v = clipNorm(vadd(v, vmul(a, cfg.dtPhys)), cfg.vMax);
          p = vadd(p, vmul(v, cfg.dtPhys));
          p[0] = Math.min(cfg.arena-1.0, Math.max(1.0, p[0]));
          p[1] = Math.min(cfg.arena-1.0, Math.max(1.0, p[1]));
          path[h] = [p[0], p[1]];

          cost += cfg.goalW * (vnorm(vsub(g, p)) / cfg.arena);

          // predicted peds
          let pedPen = 0.0;
          for(let j=0;j<cfg.M;j++){
            const pp = predictLinear(pedPos[j], pedVel[j], h);
            const d = vnorm(vsub(p, pp));
            const viol = relu(cfg.safePed - d);
            pedPen += viol*viol;
          }
          cost += cfg.riskW * (wP0 * pedPen);

          // predicted other drones (pairwise-weighted)
          let drCost = 0.0;
          for(let oi=0; oi<other.length; oi++){
            const k = other[oi];
            const op = predictLinear(world.pos[k], world.vel[k], h);
            const d = vnorm(vsub(p, op));
            const viol = relu(cfg.safeDrone - d);
            const wR = cfg.kr / Math.max(effLamPair(i, k), cfg.lamMin);
            drCost += wR * (viol*viol);
          }
          cost += cfg.riskW * drCost;
        }

        const vdir = vunit(vdes);
        cost += cfg.turnW * (1.0 - vdot(vdir, baseDir));

        if(cost < bestCost){
          bestCost = cost;
          bestV = vdes;
          bestPath = path;
        }
      }
    }
    return {vdes: bestV, path: bestPath};
  }

  function planWorld(world, controller){
    // Build active set A_t from past info (previous maximizer) and add current worst channel.
    const curr = computeLossesAndMax(world);
    const currWorst = curr.arg;
    const prevWorst = world.prevMaxChan;

    function isActivePed(i){
      return (prevWorst && prevWorst.type==="ped" && prevWorst.i===i) ||
             (currWorst.type==="ped" && currWorst.i===i);
    }
    function isActivePair(i,k){
      const a = Math.min(i,k), b = Math.max(i,k);
      const prev = prevWorst && prevWorst.type==="pair" && ((prevWorst.i===a && prevWorst.k===b) || (prevWorst.i===b && prevWorst.k===a));
      const curr = currWorst.type==="pair" && ((currWorst.i===a && currWorst.k===b) || (currWorst.i===b && currWorst.k===a));
      return prev || curr;
    }

    const lamNo = cfg.lamHi; // fixed aggressive
    const effLamPed = (i)=>{
      if(controller === "none") return lamNo;
      if(controller === "global") return world.lamG;
      return isActivePed(i) ? world.lamG : world.lamPed[i];
    };
    const effLamPair = (i,k)=>{
      if(controller === "none") return lamNo;
      if(controller === "global") return world.lamG;
      const a = Math.min(i,k), b = Math.max(i,k);
      return isActivePair(a,b) ? world.lamG : world.lamPair[a][b];
    };

    world.plan = new Array(cfg.N);
    for(let i=0;i<cfg.N;i++){
      const res = planOne(world, i, controller, effLamPed, effLamPair);
      world.vdes[i] = res.vdes;
      world.plan[i] = res.path;
    }
  }

  // ---------- Lambda updates (paper-consistent) ----------
  function updateController(world, controller){
    const {lossPed, lossPair, mt, arg} = computeLossesAndMax(world);
    // store maximizer for next step (A_{t+1} chosen from history up to t)
    world.prevMaxChan = arg;

    if(controller === "none"){
      world.avgLam = cfg.lamHi;
      return;
    }

    const eps = Math.max(cfg.epsP, cfg.epsR);
    const eta = Math.min(cfg.etaP, cfg.etaR);
    world.lamG = Math.min(cfg.lamHi, Math.max(cfg.lamLo, world.lamG + eta*(eps - mt)));

    if(controller === "global"){
      world.avgLam = world.lamG;
      return;
    }

    // local updates (Eq. 8) for efficiency
    const etaLoc = eta;
    for(let i=0;i<cfg.N;i++){
      world.lamPed[i] = Math.min(cfg.lamHi, Math.max(cfg.lamLo, world.lamPed[i] + etaLoc*(eps - lossPed[i])));
    }
    for(let i=0;i<cfg.N;i++){
      for(let k=i+1;k<cfg.N;k++){
        const l = lossPair[i][k];
        const nv = Math.min(cfg.lamHi, Math.max(cfg.lamLo, world.lamPair[i][k] + etaLoc*(eps - l)));
        world.lamPair[i][k] = nv;
        world.lamPair[k][i] = nv;
      }
    }

    // average for display
    let s=0, c=0;
    for(let i=0;i<cfg.N;i++){ s += world.lamPed[i]; c++; }
    for(let i=0;i<cfg.N;i++) for(let k=i+1;k<cfg.N;k++){ s += world.lamPair[i][k]; c++; }
    world.avgLam = s / Math.max(1,c);
  }

  // ---------- Series recording ----------
  function recordSeries(){
    for(const w of [W0,W1,W2]){
      w.dSeries.push(closestDistance(w));
      w.rSeries.push(w.reward);
      if(w.dSeries.length > 3000) w.dSeries.shift();
      if(w.rSeries.length > 3000) w.rSeries.shift();
      if(w.cSeries.length > 3000) w.cSeries.shift();
    }
  }

  // ---------- Layout ----------
  function layout(){
    const pad = 26;
    const gap = 18;
    const topH = 600;

    const panelW = (canvas.width - 2*pad - 2*gap) / 3;
    const panelH = topH - pad;

    const panels = [
      {x: pad, y: pad, w: panelW, h: panelH, world: W0, title: W0.name, col: W0.color, controller:"none"},
      {x: pad + panelW + gap, y: pad, w: panelW, h: panelH, world: W1, title: W1.name, col: W1.color, controller:"global"},
      {x: pad + 2*(panelW + gap), y: pad, w: panelW, h: panelH, world: W2, title: W2.name, col: W2.color, controller:"cswc"},
    ];

    const graphsTop = pad + panelH + 54;
    const graphsH = canvas.height - graphsTop - pad;
    const graphGap = 18;
    const eachH = (graphsH - graphGap) / 2;

    const graphD = {x: pad, y: graphsTop, w: canvas.width - 2*pad, h: eachH};
    const graphR = {x: pad, y: graphsTop + eachH + graphGap, w: canvas.width - 2*pad, h: eachH};

    return {panels, graphD, graphR};
  }

  function worldToScreen(panel, xy){
    const sx = panel.x + (xy[0] / cfg.arena) * panel.w;
    const sy = panel.y + panel.h - (xy[1] / cfg.arena) * panel.h;
    return [sx, sy];
  }

  function drawCrossScreen(x, y, size){
    ctx.beginPath();
    ctx.moveTo(x-size, y-size); ctx.lineTo(x+size, y+size);
    ctx.moveTo(x+size, y-size); ctx.lineTo(x-size, y+size);
    ctx.stroke();
  }

  function drawPanel(panel){
    const {x,y,w,h,world,col,title} = panel;

    ctx.strokeStyle = "#111"; ctx.lineWidth = 1.7;
    ctx.strokeRect(x,y,w,h);

    ctx.fillStyle = "#111";
    ctx.font = "750 13px ui-sans-serif, system-ui";
    ctx.fillText(title, x+10, y-6);

    // counters (not part of the graphs)
    ctx.fillStyle = "#111";
    ctx.font = "650 12px ui-sans-serif, system-ui";
    const counters = `rewards=${world.reward}   collisions=${world.cumColl}` +
      (panel.controller==="none" ? "" : `   λ(avg)=${world.avgLam.toFixed(2)}`);
    ctx.fillText(counters, x+10, y+18);

    // pedestrians
    for(let j=0;j<cfg.M;j++){
      const p = pedPos[j];
      const v = pedVel[j];
      const [cx, cy] = worldToScreen(panel, p);

      if(overlaysOn){
        // ring
        ctx.strokeStyle = "rgba(120,120,120,0.55)"; ctx.lineWidth = 1.6;
        const r = (cfg.safePed/cfg.arena) * w;
        ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI*2); ctx.stroke();

        // direction line
        const dir = vunit(v);
        const end = vadd(p, vmul(dir, cfg.pedDirLen));
        const [ex, ey] = worldToScreen(panel, end);
        ctx.strokeStyle = "rgba(17,17,17,0.9)"; ctx.lineWidth = 1.6;
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(ex, ey); ctx.stroke();
      }

      // pedestrian dot
      ctx.fillStyle = "rgba(110,110,110,0.75)";
      ctx.beginPath(); ctx.arc(cx, cy, 4.0, 0, Math.PI*2); ctx.fill();
    }

    // planned splines
    if(overlaysOn){
      for(let i=0;i<cfg.N;i++){
        const path = world.plan[i];
        if(!path) continue;
        const smooth = catmullRom(path, cfg.samplesPerSeg);
        ctx.strokeStyle = (panel.controller==="none") ? "rgba(31,119,180,0.55)"
          : (panel.controller==="global" ? "rgba(148,103,189,0.55)" : "rgba(255,127,14,0.55)");
        ctx.lineWidth = 3.0;
        ctx.beginPath();
        for(let k=0;k<smooth.length;k++){
          const [sx, sy] = worldToScreen(panel, smooth[k]);
          if(k===0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
        }
        ctx.stroke();
      }
    }

    // goals + drones
    for(let i=0;i<cfg.N;i++){
      const p = world.pos[i];
      const g = world.goal[i];
      const [dx, dy] = worldToScreen(panel, p);
      const [gx, gy] = worldToScreen(panel, g);

      ctx.strokeStyle = col; ctx.lineWidth = 2.0;
      drawCrossScreen(gx, gy, 6);

      ctx.fillStyle = col;
      ctx.beginPath(); ctx.arc(dx, dy, 6.0, 0, Math.PI*2); ctx.fill();
    }

    // collision crosses (temporary)
    for(const c of world.crosses){
      const [sx, sy] = worldToScreen(panel, [c.x, c.y]);
      ctx.strokeStyle = COLORS.red; ctx.lineWidth = 2.4;
      drawCrossScreen(sx, sy, 7);
    }
  }

  function drawGraphDistance(graph, worlds){
    const {x,y,w,h} = graph;

    ctx.save();
    ctx.strokeStyle = "#111"; ctx.lineWidth = 1.7;
    ctx.strokeRect(x,y,w,h);

    ctx.fillStyle = "#111";
    ctx.font = "750 13px ui-sans-serif, system-ui";
    ctx.fillText("Closest distance over time (collision markers ×)", x+10, y-6);

    const padL = 54, padR = 18, padT = 24, padB = 34;
    const gx0 = x + padL, gy0 = y + padT;
    const gw = w - padL - padR, gh = h - padT - padB;

    const maxPts = 900;
    const seriesLen = Math.max(...worlds.map(w => w.dSeries.length));
    const i0 = Math.max(0, seriesLen - maxPts);

    const dMin = 0.0;
    let dMax = 6.0;
    for(const wld of worlds){
      for(let i=i0;i<wld.dSeries.length;i++) dMax = Math.max(dMax, wld.dSeries[i]);
    }
    dMax = Math.min(12.0, dMax + 0.5);

    const t0 = i0 * cfg.dtPhys;
    const t1 = Math.max(t0 + 1e-6, (seriesLen-1) * cfg.dtPhys);

    const X = (t)=> gx0 + (t - t0)/(t1 - t0) * gw;
    const Y = (d)=> gy0 + gh - (d - dMin)/(dMax - dMin) * gh;

    // grid + labels
    ctx.strokeStyle = "rgba(0,0,0,0.08)"; ctx.lineWidth = 1;
    const yTicks = 5;
    for(let k=0;k<=yTicks;k++){
      const d = dMin + (k/yTicks)*(dMax - dMin);
      const yy = Y(d);
      ctx.beginPath(); ctx.moveTo(gx0, yy); ctx.lineTo(gx0+gw, yy); ctx.stroke();
      ctx.fillStyle = "rgba(0,0,0,0.65)";
      ctx.font = "600 11px ui-sans-serif, system-ui";
      ctx.fillText(d.toFixed(1), x+10, yy+4);
    }
    const xTicks = 6;
    for(let k=0;k<=xTicks;k++){
      const t = t0 + (k/xTicks)*(t1 - t0);
      const xx = X(t);
      ctx.beginPath(); ctx.moveTo(xx, gy0); ctx.lineTo(xx, gy0+gh); ctx.stroke();
      ctx.fillStyle = "rgba(0,0,0,0.65)";
      ctx.font = "600 11px ui-sans-serif, system-ui";
      ctx.fillText(t.toFixed(1)+"s", xx-14, y+h-10);
    }

    function drawSeries(wld, color){
      const arr = wld.dSeries;
      if(arr.length < 2) return;
      ctx.strokeStyle = color; ctx.lineWidth = 2.4;
      ctx.beginPath();
      for(let i=i0;i<arr.length;i++){
        const t = i*cfg.dtPhys;
        const xx = X(t);
        const yy = Y(arr[i]);
        if(i===i0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
      }
      ctx.stroke();

      // collision markers
      ctx.strokeStyle = COLORS.red; ctx.lineWidth = 2.0;
      for(const c of wld.cSeries){
        if(c.t < t0 || c.t > t1) continue;
        const xx = X(c.t);
        const yy = Y(Math.max(dMin, Math.min(dMax, c.d)));
        ctx.beginPath();
        ctx.moveTo(xx-5, yy-5); ctx.lineTo(xx+5, yy+5);
        ctx.moveTo(xx+5, yy-5); ctx.lineTo(xx-5, yy+5);
        ctx.stroke();
      }
    }

    drawSeries(W0, COLORS.blue);
    drawSeries(W1, COLORS.purple);
    drawSeries(W2, COLORS.orange);

    ctx.restore();
  }

  function drawGraphReward(graph, worlds){
    const {x,y,w,h} = graph;

    ctx.save();
    ctx.strokeStyle = "#111"; ctx.lineWidth = 1.7;
    ctx.strokeRect(x,y,w,h);

    ctx.fillStyle = "#111";
    ctx.font = "750 13px ui-sans-serif, system-ui";
    ctx.fillText("Reward over time (cumulative)", x+10, y-6);

    const padL = 54, padR = 18, padT = 24, padB = 34;
    const gx0 = x + padL, gy0 = y + padT;
    const gw = w - padL - padR, gh = h - padT - padB;

    const maxPts = 900;
    const seriesLen = Math.max(...worlds.map(w => w.rSeries.length));
    const i0 = Math.max(0, seriesLen - maxPts);

    const rMin = 0;
    let rMax = 5;
    for(const wld of worlds){
      for(let i=i0;i<wld.rSeries.length;i++) rMax = Math.max(rMax, wld.rSeries[i]);
    }
    rMax = Math.max(rMax, rMin + 1);

    const t0 = i0 * cfg.dtPhys;
    const t1 = Math.max(t0 + 1e-6, (seriesLen-1) * cfg.dtPhys);

    const X = (t)=> gx0 + (t - t0)/(t1 - t0) * gw;
    const Y = (r)=> gy0 + gh - (r - rMin)/(rMax - rMin) * gh;

    // grid + labels
    ctx.strokeStyle = "rgba(0,0,0,0.08)"; ctx.lineWidth = 1;
    const yTicks = 5;
    for(let k=0;k<=yTicks;k++){
      const r = rMin + (k/yTicks)*(rMax - rMin);
      const yy = Y(r);
      ctx.beginPath(); ctx.moveTo(gx0, yy); ctx.lineTo(gx0+gw, yy); ctx.stroke();
      ctx.fillStyle = "rgba(0,0,0,0.65)";
      ctx.font = "600 11px ui-sans-serif, system-ui";
      ctx.fillText(String(Math.round(r)), x+10, yy+4);
    }
    const xTicks = 6;
    for(let k=0;k<=xTicks;k++){
      const t = t0 + (k/xTicks)*(t1 - t0);
      const xx = X(t);
      ctx.beginPath(); ctx.moveTo(xx, gy0); ctx.lineTo(xx, gy0+gh); ctx.stroke();
      ctx.fillStyle = "rgba(0,0,0,0.65)";
      ctx.font = "600 11px ui-sans-serif, system-ui";
      ctx.fillText(t.toFixed(1)+"s", xx-14, y+h-10);
    }

    function drawSeries(wld, color){
      const arr = wld.rSeries;
      if(arr.length < 2) return;
      ctx.strokeStyle = color; ctx.lineWidth = 2.4;
      ctx.beginPath();
      for(let i=i0;i<arr.length;i++){
        const t = i*cfg.dtPhys;
        const xx = X(t);
        const yy = Y(arr[i]);
        if(i===i0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
      }
      ctx.stroke();
    }

    drawSeries(W0, COLORS.blue);
    drawSeries(W1, COLORS.purple);
    drawSeries(W2, COLORS.orange);

    ctx.restore();
  }

  function draw(){
    ctx.clearRect(0,0,canvas.width,canvas.height);
    const {panels, graphD, graphR} = layout();
    for(const p of panels) drawPanel(p);
    drawGraphDistance(graphD, [W0,W1,W2]);
    drawGraphReward(graphR, [W0,W1,W2]);
  }

  // ---------- Loop ----------
  let last = performance.now();
  let physAccum = 0;
  let planAccum = 0;

  function tick(now){
    const dt = (now - last) / 1000;
    last = now;

    if(!paused){
      physAccum += dt;
      planAccum += dt;
      const planPeriod = 1 / Math.max(1, cfg.planHz);

      // planning
      while(planAccum >= planPeriod){
        // resize plan arrays if H changed
        for(const w of [W0,W1,W2]){
          if(w.plan.length !== cfg.N || (w.plan[0] && w.plan[0].length !== cfg.H+1)){
            w.plan = new Array(cfg.N).fill(0).map(()=>new Array(cfg.H+1).fill(0).map(()=>[0,0]));
          }
        }
        planWorld(W0, "none");
        planWorld(W1, "global");
        planWorld(W2, "cswc");
        planAccum -= planPeriod;
      }

      // physics
      while(physAccum >= cfg.dtPhys){
        stepPeds();

        inertialStep(W0);
        inertialStep(W1);
        inertialStep(W2);

        updateGoalsAndReward(W0);
        updateGoalsAndReward(W1);
        updateGoalsAndReward(W2);

        handleCollisions(W0);
        handleCollisions(W1);
        handleCollisions(W2);

        updateController(W0, "none");
        updateController(W1, "global");
        updateController(W2, "cswc");

        recordSeries();

        tSim += cfg.dtPhys;
        physAccum -= cfg.dtPhys;
      }
    }

    draw();
    requestAnimationFrame(tick);
  }

  // ---------- Start ----------
  initWorlds();
  planWorld(W0, "none");
  planWorld(W1, "global");
  planWorld(W2, "cswc");
  requestAnimationFrame(tick);
})();
