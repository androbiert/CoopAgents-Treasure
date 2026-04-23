/**
 * MARL Q-Mix Explorer — Interactive GridWorld Visualizer
 * Renders cooperative agent trajectories with smooth animations,
 * trail effects, episode browsing, and full playback controls.
 */
document.addEventListener('DOMContentLoaded', async () => {
    // ─── State ───────────────────────────────────────────
    let trajectory = [];
    let currentStep = 0;
    let isPlaying = false;
    let playInterval = null;
    let playSpeed = 600;

    // Episode browsing state
    let dataSource = 'training'; // 'training' or 'eval'
    let trainingData = null;     // Full training_data.json
    let evalTrajectory = null;   // trajectory.json
    let currentEpisodeIndex = 0;
    let episodes = [];           // Array of episode objects from training data

    // ─── DOM refs ────────────────────────────────────────
    const gridContainer  = document.getElementById('grid-container');
    const valStep        = document.getElementById('val-step');
    const valReward      = document.getElementById('val-reward');
    const valDoor        = document.getElementById('val-door');
    const valPlate       = document.getElementById('val-plate');
    const posA           = document.getElementById('pos-a');
    const posB           = document.getElementById('pos-b');
    const actionA        = document.getElementById('action-a');
    const actionB        = document.getElementById('action-b');
    const btnPlay        = document.getElementById('btn-play');
    const btnPrev        = document.getElementById('btn-prev');
    const btnNext        = document.getElementById('btn-next');
    const btnReset       = document.getElementById('btn-reset');
    const timeline       = document.getElementById('timeline');
    const tlCurrent      = document.getElementById('tl-current');
    const tlTotal        = document.getElementById('tl-total');
    const iconPlay       = document.getElementById('icon-play');
    const iconPause      = document.getElementById('icon-pause');
    const statusChip     = document.getElementById('status-chip');
    const statusText     = statusChip.querySelector('.status-text');
    const speedSelect    = document.getElementById('speed-select');
    const coordLabelsX   = document.getElementById('coord-labels-x');
    const coordLabelsY   = document.getElementById('coord-labels-y');
    const envSize        = document.getElementById('env-size');
    const envWall        = document.getElementById('env-wall');
    const envDoorY       = document.getElementById('env-door-y');

    // Episode bar DOM refs
    const srcTraining    = document.getElementById('src-training');
    const srcEval        = document.getElementById('src-eval');
    const episodeCounter = document.getElementById('episode-counter');
    const epFirst        = document.getElementById('ep-first');
    const epPrev         = document.getElementById('ep-prev');
    const epNext         = document.getElementById('ep-next');
    const epLast         = document.getElementById('ep-last');
    const epNum          = document.getElementById('ep-num');
    const epRewardBadge  = document.getElementById('ep-reward-badge');
    const epStepsBadge   = document.getElementById('ep-steps-badge');
    const epSuccessBadge = document.getElementById('ep-success-badge');
    const epEpsilonBadge = document.getElementById('ep-epsilon-badge');

    // ─── Action names ────────────────────────────────────
    const ACTION_NAMES = ['↑ Haut', '↓ Bas', '← Gauche', '→ Droite', '· Stop'];

    // ─── Load data ───────────────────────────────────────

    // Try loading training_data.json
    try {
        const resp = await fetch('training_data.json');
        if (resp.ok) {
            trainingData = await resp.json();
            episodes = trainingData.episodes || [];
        }
    } catch (e) {
        console.warn('training_data.json not available.');
    }

    // Try loading trajectory.json (evaluation)
    try {
        const resp = await fetch('trajectory.json');
        if (resp.ok) {
            evalTrajectory = await resp.json();
        }
    } catch (e) {
        console.warn('trajectory.json not available.');
    }

    // Determine initial data source
    if (episodes.length > 0) {
        dataSource = 'training';
        trajectory = episodes[0].trajectory;
    } else if (evalTrajectory && evalTrajectory.length > 0) {
        dataSource = 'eval';
        trajectory = evalTrajectory;
    } else {
        // Use demo
        trajectory = generateDemoTrajectory();
        dataSource = 'eval';
        evalTrajectory = trajectory;
    }

    // ─── Source selector UI sync ─────────────────────────
    function updateSourceButtons() {
        srcTraining.classList.toggle('active', dataSource === 'training');
        srcEval.classList.toggle('active', dataSource === 'eval');
    }

    function switchSource(source) {
        if (isPlaying) setPlaying(false);
        dataSource = source;
        updateSourceButtons();

        if (source === 'training' && episodes.length > 0) {
            currentEpisodeIndex = 0;
            trajectory = episodes[0].trajectory;
        } else if (source === 'eval' && evalTrajectory) {
            trajectory = evalTrajectory;
        } else {
            trajectory = generateDemoTrajectory();
        }

        currentStep = 0;
        prevAgentAPos = null;
        prevAgentBPos = null;
        updateEpisodeInfo();
        updateTimeline();
        buildGrid();
        renderStep(0);
    }

    srcTraining.addEventListener('click', () => switchSource('training'));
    srcEval.addEventListener('click', () => switchSource('eval'));

    // ─── Episode navigation ──────────────────────────────
    function updateEpisodeInfo() {
        if (dataSource === 'training' && episodes.length > 0) {
            const ep = episodes[currentEpisodeIndex];
            episodeCounter.textContent = `${currentEpisodeIndex + 1} / ${episodes.length}`;
            epNum.textContent = `Ép. ${ep.episode}`;

            // Reward badge
            const rw = ep.total_reward;
            epRewardBadge.textContent = `R: ${rw >= 0 ? '+' : ''}${rw.toFixed(1)}`;
            epRewardBadge.className = 'ep-badge ep-badge-reward ' + (rw >= 5 ? 'positive' : rw < 0 ? 'negative' : '');

            // Steps
            epStepsBadge.textContent = `${ep.steps} pas`;

            // Success
            if (ep.success) {
                epSuccessBadge.textContent = '✓ Succès';
                epSuccessBadge.className = 'ep-badge ep-badge-success win';
            } else {
                epSuccessBadge.textContent = '✗ Échec';
                epSuccessBadge.className = 'ep-badge ep-badge-success fail';
            }

            // Epsilon
            epEpsilonBadge.textContent = `ε: ${ep.epsilon.toFixed(3)}`;
        } else if (dataSource === 'eval') {
            episodeCounter.textContent = '1 / 1';
            epNum.textContent = 'Évaluation';
            epRewardBadge.textContent = 'R: —';
            epRewardBadge.className = 'ep-badge ep-badge-reward';
            epStepsBadge.textContent = `${trajectory.length} pas`;
            epSuccessBadge.textContent = '—';
            epSuccessBadge.className = 'ep-badge ep-badge-success';
            epEpsilonBadge.textContent = 'ε: 0';
        } else {
            episodeCounter.textContent = '— / —';
            epNum.textContent = 'Démo';
            epRewardBadge.textContent = 'R: —';
            epRewardBadge.className = 'ep-badge ep-badge-reward';
            epStepsBadge.textContent = '—';
            epSuccessBadge.textContent = '—';
            epSuccessBadge.className = 'ep-badge ep-badge-success';
            epEpsilonBadge.textContent = 'ε: —';
        }
    }

    function goToEpisode(index) {
        if (dataSource !== 'training' || episodes.length === 0) return;
        if (index < 0 || index >= episodes.length) return;
        if (isPlaying) setPlaying(false);

        currentEpisodeIndex = index;
        trajectory = episodes[index].trajectory;
        currentStep = 0;
        prevAgentAPos = null;
        prevAgentBPos = null;
        updateEpisodeInfo();
        updateTimeline();
        buildGrid();
        renderStep(0);
    }

    epFirst.addEventListener('click', () => goToEpisode(0));
    epPrev.addEventListener('click', () => goToEpisode(currentEpisodeIndex - 1));
    epNext.addEventListener('click', () => goToEpisode(currentEpisodeIndex + 1));
    epLast.addEventListener('click', () => goToEpisode(episodes.length - 1));

    function updateTimeline() {
        timeline.max = trajectory.length - 1;
        tlTotal.textContent = trajectory.length - 1;
        timeline.value = 0;
        tlCurrent.textContent = 0;
    }

    // ─── Grid Setup ──────────────────────────────────────
    if (trajectory.length === 0) return;

    const conf = trajectory[0];
    let W = conf.width;
    let H = conf.height;

    // Responsive cell sizing
    const viewportEl = document.getElementById('viewport');
    function computeCellSize() {
        const vw = viewportEl.clientWidth - 80;
        const vh = viewportEl.clientHeight - 220; // extra space for episode bar
        return Math.min(Math.floor(vw / W), Math.floor(vh / H), 72);
    }
    let CELL = computeCellSize();

    // Update env info panel
    envSize.textContent = `${W} × ${H}`;
    envWall.textContent = conf.wall_col;
    envDoorY.textContent = conf.door_y;

    // ─── Build Grid ──────────────────────────────────────
    function buildGrid() {
        // Recompute from current trajectory
        if (trajectory.length === 0) return;
        const c = trajectory[0];
        W = c.width;
        H = c.height;

        CELL = computeCellSize();
        gridContainer.innerHTML = '';
        gridContainer.style.width = `${W * CELL}px`;
        gridContainer.style.height = `${H * CELL}px`;
        gridContainer.style.gridTemplateColumns = `repeat(${W}, 1fr)`;
        gridContainer.style.gridTemplateRows = `repeat(${H}, 1fr)`;

        // Cells
        for (let row = 0; row < H; row++) {
            for (let col = 0; col < W; col++) {
                const cell = document.createElement('div');
                cell.className = 'cell';
                cell.dataset.x = col;
                cell.dataset.y = H - 1 - row;
                gridContainer.appendChild(cell);
            }
        }

        // Walls
        for (let y = 0; y < H; y++) {
            if (y !== c.door_y) {
                const wall = document.createElement('div');
                wall.className = 'wall-block';
                wall.style.left   = `${c.wall_col * CELL}px`;
                wall.style.top    = `${(H - 1 - y) * CELL}px`;
                wall.style.width  = `${CELL}px`;
                wall.style.height = `${CELL}px`;
                const wallImg = document.createElement('img');
                wallImg.src = '/images/blockwall.png';
                wallImg.alt = 'Mur';
                wallImg.className = 'wall-sprite';
                wall.appendChild(wallImg);
                gridContainer.appendChild(wall);
            }
        }

        // Treasure
        const treasure = document.createElement('div');
        treasure.className = 'treasure-entity';
        treasure.id = 'treasure';
        const tSize = CELL * 0.82;
        treasure.style.width  = `${tSize}px`;
        treasure.style.height = `${tSize}px`;
        treasure.style.left   = `${c.treasure_pos[0] * CELL + (CELL - tSize) / 2}px`;
        treasure.style.top    = `${(H - 1 - c.treasure_pos[1]) * CELL + (CELL - tSize) / 2}px`;
        const treasureImg = document.createElement('img');
        treasureImg.src = '/images/tresor.png';
        treasureImg.alt = 'Trésor';
        treasureImg.className = 'entity-sprite';
        treasure.appendChild(treasureImg);
        gridContainer.appendChild(treasure);

        // Plate
        const plate = document.createElement('div');
        plate.className = 'plate-entity';
        plate.id = 'plate';
        const pSize = CELL * 0.78;
        plate.style.width  = `${pSize}px`;
        plate.style.height = `${pSize}px`;
        plate.style.left   = `${c.plate_pos[0] * CELL + (CELL - pSize) / 2}px`;
        plate.style.top    = `${(H - 1 - c.plate_pos[1]) * CELL + (CELL - pSize) / 2}px`;
        gridContainer.appendChild(plate);

        // Door
        const door = document.createElement('div');
        door.className = 'door-entity';
        door.id = 'door';
        const dSize = CELL * 0.92;
        door.style.left   = `${c.wall_col * CELL + (CELL - dSize) / 2}px`;
        door.style.top    = `${(H - 1 - c.door_y) * CELL + (CELL - dSize) / 2}px`;
        door.style.width  = `${dSize}px`;
        door.style.height = `${dSize}px`;
        const doorImgClosed = document.createElement('img');
        doorImgClosed.src = '/images/closed.png';
        doorImgClosed.alt = 'Porte fermée';
        doorImgClosed.className = 'door-sprite door-closed-img';
        door.appendChild(doorImgClosed);
        const doorImgOpen = document.createElement('img');
        doorImgOpen.src = '/images/opened.png';
        doorImgOpen.alt = 'Porte ouverte';
        doorImgOpen.className = 'door-sprite door-open-img';
        doorImgOpen.style.display = 'none';
        door.appendChild(doorImgOpen);
        gridContainer.appendChild(door);

        // Agents
        const aSize = CELL * 0.85;
        const agentAEl = document.createElement('div');
        agentAEl.className = 'entity agent-a';
        agentAEl.id = 'agent-a';
        agentAEl.style.width  = `${aSize}px`;
        agentAEl.style.height = `${aSize}px`;
        const imgA = document.createElement('img');
        imgA.src = '/images/Blue_agent.png';
        imgA.alt = 'Agent Alpha';
        imgA.className = 'entity-sprite';
        agentAEl.appendChild(imgA);
        gridContainer.appendChild(agentAEl);

        const agentBEl = document.createElement('div');
        agentBEl.className = 'entity agent-b';
        agentBEl.id = 'agent-b';
        agentBEl.style.width  = `${aSize}px`;
        agentBEl.style.height = `${aSize}px`;
        const imgB = document.createElement('img');
        imgB.src = '/images/red_agent.png';
        imgB.alt = 'Agent Beta';
        imgB.className = 'entity-sprite';
        agentBEl.appendChild(imgB);
        gridContainer.appendChild(agentBEl);

        // Coordinate labels
        coordLabelsX.innerHTML = '';
        coordLabelsY.innerHTML = '';
        for (let x = 0; x < W; x++) {
            const lbl = document.createElement('span');
            lbl.className = 'coord-label';
            lbl.textContent = x;
            coordLabelsX.appendChild(lbl);
        }
        for (let y = 0; y < H; y++) {
            const lbl = document.createElement('span');
            lbl.className = 'coord-label';
            lbl.textContent = y;
            coordLabelsY.appendChild(lbl);
        }
    }

    buildGrid();
    updateTimeline();
    updateSourceButtons();
    updateEpisodeInfo();

    // ─── Render a Step ───────────────────────────────────
    let prevAgentAPos = null;
    let prevAgentBPos = null;

    function renderStep(index) {
        if (index >= trajectory.length) return;
        const state = trajectory[index];
        const agentAEl = document.getElementById('agent-a');
        const agentBEl = document.getElementById('agent-b');
        const doorEl   = document.getElementById('door');
        const plateEl  = document.getElementById('plate');

        if (!agentAEl || !agentBEl || !doorEl || !plateEl) return;

        const aSize = CELL * 0.85;

        // Agent positions
        const aLeft = state.agent_0[0] * CELL + (CELL - aSize) / 2;
        const aTop  = (H - 1 - state.agent_0[1]) * CELL + (CELL - aSize) / 2;
        const bLeft = state.agent_1[0] * CELL + (CELL - aSize) / 2;
        const bTop  = (H - 1 - state.agent_1[1]) * CELL + (CELL - aSize) / 2;

        // Trail effect
        if (prevAgentAPos && (prevAgentAPos.left !== aLeft || prevAgentAPos.top !== aTop)) {
            createTrail(prevAgentAPos.left, prevAgentAPos.top, aSize, 'trail-a');
        }
        if (prevAgentBPos && (prevAgentBPos.left !== bLeft || prevAgentBPos.top !== bTop)) {
            createTrail(prevAgentBPos.left, prevAgentBPos.top, aSize, 'trail-b');
        }

        prevAgentAPos = { left: aLeft, top: aTop };
        prevAgentBPos = { left: bLeft, top: bTop };

        agentAEl.style.left = `${aLeft}px`;
        agentAEl.style.top  = `${aTop}px`;
        agentBEl.style.left = `${bLeft}px`;
        agentBEl.style.top  = `${bTop}px`;

        // Door
        if (state.door_open) {
            doorEl.classList.add('open');
            plateEl.classList.add('active');
            // Swap door images
            const closedImg = doorEl.querySelector('.door-closed-img');
            const openImg = doorEl.querySelector('.door-open-img');
            if (closedImg) closedImg.style.display = 'none';
            if (openImg) openImg.style.display = 'block';
            valDoor.innerHTML = '<span class="door-badge open">Ouverte</span>';
            valPlate.innerHTML = '<span class="plate-badge active">Active</span>';
        } else {
            doorEl.classList.remove('open');
            plateEl.classList.remove('active');
            // Swap door images
            const closedImg = doorEl.querySelector('.door-closed-img');
            const openImg = doorEl.querySelector('.door-open-img');
            if (closedImg) closedImg.style.display = 'block';
            if (openImg) openImg.style.display = 'none';
            valDoor.innerHTML = '<span class="door-badge closed">Fermée</span>';
            valPlate.innerHTML = '<span class="plate-badge inactive">Inactive</span>';
        }

        // Metrics
        valStep.textContent = state.step;
        valReward.textContent = state.reward >= 0 ? `+${state.reward.toFixed(1)}` : state.reward.toFixed(1);
        valReward.style.color = state.reward >= 5 ? '#22c55e' : state.reward < 0 ? '#f87171' : '#eef0f6';

        // Agent positions in sidebar
        posA.textContent = `(${state.agent_0[0]}, ${state.agent_0[1]})`;
        posB.textContent = `(${state.agent_1[0]}, ${state.agent_1[1]})`;

        // Infer actions between steps
        if (index > 0) {
            const prev = trajectory[index - 1];
            actionA.textContent = inferAction(prev.agent_0, state.agent_0);
            actionB.textContent = inferAction(prev.agent_1, state.agent_1);
        } else {
            actionA.textContent = '—';
            actionB.textContent = '—';
        }

        // Timeline
        timeline.value = index;
        tlCurrent.textContent = index;

        // Success flash
        if (state.reward >= 5 && index > 0 && trajectory[index - 1].reward < 5) {
            gridContainer.classList.remove('success');
            void gridContainer.offsetWidth;
            gridContainer.classList.add('success');
        }
    }

    function createTrail(left, top, size, className) {
        const trail = document.createElement('div');
        trail.className = `agent-trail ${className}`;
        trail.style.left   = `${left}px`;
        trail.style.top    = `${top}px`;
        trail.style.width  = `${size}px`;
        trail.style.height = `${size}px`;
        gridContainer.appendChild(trail);
        trail.addEventListener('animationend', () => trail.remove());
    }

    function inferAction(prevPos, currPos) {
        const dx = currPos[0] - prevPos[0];
        const dy = currPos[1] - prevPos[1];
        if (dy > 0) return ACTION_NAMES[0];
        if (dy < 0) return ACTION_NAMES[1];
        if (dx < 0) return ACTION_NAMES[2];
        if (dx > 0) return ACTION_NAMES[3];
        return ACTION_NAMES[4];
    }

    // ─── Playback Controls ───────────────────────────────
    function setPlaying(playing) {
        isPlaying = playing;
        iconPlay.style.display  = playing ? 'none' : 'block';
        iconPause.style.display = playing ? 'block' : 'none';

        if (playing) {
            statusChip.classList.add('playing');
            statusText.textContent = 'Lecture';
            if (currentStep >= trajectory.length - 1) currentStep = 0;
            playInterval = setInterval(() => {
                if (currentStep < trajectory.length - 1) {
                    currentStep++;
                    renderStep(currentStep);
                } else {
                    setPlaying(false);
                    statusText.textContent = 'Terminé';
                }
            }, playSpeed);
        } else {
            statusChip.classList.remove('playing');
            if (statusText.textContent !== 'Terminé') statusText.textContent = 'Prêt';
            clearInterval(playInterval);
            playInterval = null;
        }
    }

    btnPlay.addEventListener('click', () => setPlaying(!isPlaying));

    btnNext.addEventListener('click', () => {
        if (isPlaying) setPlaying(false);
        if (currentStep < trajectory.length - 1) {
            currentStep++;
            renderStep(currentStep);
        }
    });

    btnPrev.addEventListener('click', () => {
        if (isPlaying) setPlaying(false);
        if (currentStep > 0) {
            currentStep--;
            renderStep(currentStep);
        }
    });

    btnReset.addEventListener('click', () => {
        if (isPlaying) setPlaying(false);
        currentStep = 0;
        prevAgentAPos = null;
        prevAgentBPos = null;
        renderStep(0);
        statusText.textContent = 'Prêt';
    });

    timeline.addEventListener('input', (e) => {
        if (isPlaying) setPlaying(false);
        currentStep = parseInt(e.target.value);
        prevAgentAPos = null;
        prevAgentBPos = null;
        renderStep(currentStep);
    });

    speedSelect.addEventListener('change', (e) => {
        playSpeed = parseInt(e.target.value);
        if (isPlaying) {
            setPlaying(false);
            setPlaying(true);
        }
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.key === ' ' || e.key === 'k') {
            e.preventDefault();
            setPlaying(!isPlaying);
        }
        if (e.key === 'ArrowRight' || e.key === 'l') {
            e.preventDefault();
            if (isPlaying) setPlaying(false);
            if (currentStep < trajectory.length - 1) {
                currentStep++;
                renderStep(currentStep);
            }
        }
        if (e.key === 'ArrowLeft' || e.key === 'j') {
            e.preventDefault();
            if (isPlaying) setPlaying(false);
            if (currentStep > 0) {
                currentStep--;
                renderStep(currentStep);
            }
        }
        if (e.key === 'r') {
            btnReset.click();
        }
        // Episode navigation: PageUp / PageDown
        if (e.key === 'PageDown' && dataSource === 'training') {
            e.preventDefault();
            goToEpisode(currentEpisodeIndex + 1);
        }
        if (e.key === 'PageUp' && dataSource === 'training') {
            e.preventDefault();
            goToEpisode(currentEpisodeIndex - 1);
        }
    });

    // ─── Responsive resize ───────────────────────────────
    let resizeTimer;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
            buildGrid();
            prevAgentAPos = null;
            prevAgentBPos = null;
            renderStep(currentStep);
        }, 200);
    });

    // ─── Initial Render ──────────────────────────────────
    renderStep(0);

    // ─── Auto-Refresh: Poll for new training data ────────
    let autoRefreshInterval = null;
    let lastEpisodeCount = episodes.length;
    // Remember the first episode number to detect a new training session
    let lastFirstEpisodeNum = (episodes.length > 0) ? episodes[0].episode : -1;

    function reloadAllData(newData) {
        trainingData = newData;
        episodes = newData.episodes || [];
        lastEpisodeCount = episodes.length;
        lastFirstEpisodeNum = (episodes.length > 0) ? episodes[0].episode : -1;

        dataSource = 'training';
        updateSourceButtons();
        currentEpisodeIndex = episodes.length - 1;
        trajectory = episodes[currentEpisodeIndex].trajectory;
        currentStep = 0;
        prevAgentAPos = null;
        prevAgentBPos = null;
        updateTimeline();
        buildGrid();
        renderStep(0);
        updateEpisodeInfo();
    }

    function startAutoRefresh() {
        if (autoRefreshInterval) return;
        autoRefreshInterval = setInterval(async () => {
            try {
                const resp = await fetch('training_data.json?t=' + Date.now());
                if (!resp.ok) return;
                const newData = await resp.json();
                const newEpisodes = newData.episodes || [];

                // ── Detect a NEW training session (episode numbers reset) ──
                const newFirstEpisodeNum = (newEpisodes.length > 0) ? newEpisodes[0].episode : -1;
                const isNewSession = (newFirstEpisodeNum !== lastFirstEpisodeNum) ||
                                     (newData.total_episodes < (trainingData ? trainingData.total_episodes : 0));

                if (isNewSession) {
                    // Fresh training started — reload everything
                    reloadAllData(newData);
                    statusText.textContent = `Nouveau · ${newData.total_episodes} éps`;
                    statusChip.classList.add('playing');
                    return;
                }

                if (newEpisodes.length > lastEpisodeCount) {
                    trainingData = newData;
                    episodes = newEpisodes;

                    // Auto-switch to training source if we were on demo
                    if (dataSource !== 'training' && lastEpisodeCount === 0) {
                        dataSource = 'training';
                        updateSourceButtons();
                    }

                    // If user is on training view, auto-jump to latest episode
                    if (dataSource === 'training') {
                        const wasAtLast = (currentEpisodeIndex === lastEpisodeCount - 1) || lastEpisodeCount === 0;
                        if (wasAtLast) {
                            currentEpisodeIndex = episodes.length - 1;
                            trajectory = episodes[currentEpisodeIndex].trajectory;
                            currentStep = 0;
                            prevAgentAPos = null;
                            prevAgentBPos = null;
                            updateTimeline();
                            buildGrid();
                            renderStep(0);
                        }
                        updateEpisodeInfo();
                    }

                    lastEpisodeCount = newEpisodes.length;

                    // Update status to show live training
                    statusText.textContent = `Live · ${newData.total_episodes} éps`;
                    statusChip.classList.add('playing');
                }

                // Detect training complete
                if (newData.total_episodes >= newData.n_episodes_target) {
                    statusText.textContent = 'Entraînement terminé ✓';
                    stopAutoRefresh();
                }
            } catch (e) {
                // File not ready yet, silently ignore
            }
        }, 5000); // Poll every 5 seconds
    }

    function stopAutoRefresh() {
        if (autoRefreshInterval) {
            clearInterval(autoRefreshInterval);
            autoRefreshInterval = null;
        }
    }

    // Start polling immediately
    startAutoRefresh();

    // ─── Demo Trajectory Generator ───────────────────────
    function generateDemoTrajectory() {
        const base = { width: 7, height: 7, wall_col: 3, door_y: 3, plate_pos: [1, 1], treasure_pos: [5, 5] };
        const steps = [
            { agent_0: [0, 0], agent_1: [0, 6], door_open: false },
            { agent_0: [1, 0], agent_1: [0, 5], door_open: false },
            { agent_0: [1, 1], agent_1: [0, 4], door_open: true  },
            { agent_0: [1, 1], agent_1: [0, 3], door_open: true  },
            { agent_0: [1, 1], agent_1: [1, 3], door_open: true  },
            { agent_0: [1, 1], agent_1: [2, 3], door_open: true  },
            { agent_0: [1, 1], agent_1: [3, 3], door_open: true  },
            { agent_0: [1, 1], agent_1: [4, 3], door_open: true  },
            { agent_0: [1, 1], agent_1: [4, 4], door_open: true  },
            { agent_0: [1, 1], agent_1: [5, 4], door_open: true  },
            { agent_0: [1, 1], agent_1: [5, 5], door_open: true  },
        ];
        let cumReward = 0;
        return steps.map((s, i) => {
            if (i > 0) cumReward -= 0.1;
            if (i === steps.length - 1) cumReward += 10;
            return {
                step: i,
                agent_0: s.agent_0,
                agent_1: s.agent_1,
                door_open: s.door_open,
                reward: parseFloat(cumReward.toFixed(1)),
                ...base
            };
        });
    }
});
