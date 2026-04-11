#!/usr/bin/env node
"use strict";

const childProcess = require("child_process");
const fs = require("fs");
const net = require("net");
const os = require("os");
const path = require("path");
const readline = require("readline");

const DEFAULTS = {
  host: "127.0.0.1",
  port: 8765,
  pollMs: 40,
  axisDeadband: 0.10,
  baseLinearSpeedMps: 0.60,
  maxLinearSpeedMps: 1.50,
  maxSteeringDeg: 18.0,
  wheelbaseM: 0.30,
  steeringStick: "right",
  holdForwardRampDelayS: 0.80,
  holdForwardRampDurationS: 3.00,
  holdForwardActivationAxis: 0.20,
  sessionToggleTriggerThreshold: 0.65,
  requireAButton: false,
};

function printHelpAndExit(code) {
  console.log(`Usage: apex_xbox_bridge.exe [options]

Options:
  --host <addr>            WSL bridge host. Default: 127.0.0.1
  --port <num>             WSL bridge port. Default: 8765
  --poll-ms <num>          Xbox poll period. Default: 40
  --base-linear-speed <m/s> Base forward speed before hold boost. Default: 0.60
  --max-linear-speed <m/s> Forward speed ceiling after hold boost. Default: 1.50
  --max-steering-deg <deg> Steering limit. Default: 18
  --wheelbase-m <m>        Wheelbase used for angular_z. Default: 0.30
  --steering-stick <s>     left | right. Default: right
  --hold-ramp-delay-s <s>  Time holding forward before boost starts. Default: 0.80
  --hold-ramp-duration-s <s> Time to ramp from base to max. Default: 3.00
  --session-toggle-trigger-threshold <0..1> Trigger threshold for session toggle. Default: 0.65
  --require-a-button       Require A button to enable movement
  -h, --help               Show this help
`);
  process.exit(code);
}

function parseArgs(argv) {
  const cfg = { ...DEFAULTS };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = () => argv[++i];
    if (arg === "--host") {
      cfg.host = next() || cfg.host;
    } else if (arg === "--port") {
      cfg.port = Number(next() || cfg.port);
    } else if (arg === "--poll-ms") {
      cfg.pollMs = Number(next() || cfg.pollMs);
    } else if (arg === "--base-linear-speed") {
      cfg.baseLinearSpeedMps = Number(next() || cfg.baseLinearSpeedMps);
    } else if (arg === "--max-linear-speed") {
      cfg.maxLinearSpeedMps = Number(next() || cfg.maxLinearSpeedMps);
    } else if (arg === "--max-steering-deg") {
      cfg.maxSteeringDeg = Number(next() || cfg.maxSteeringDeg);
    } else if (arg === "--wheelbase-m") {
      cfg.wheelbaseM = Number(next() || cfg.wheelbaseM);
    } else if (arg === "--steering-stick") {
      const stick = String(next() || cfg.steeringStick).trim().toLowerCase();
      if (stick === "left" || stick === "right") {
        cfg.steeringStick = stick;
      } else {
        console.error(`[APEX][xbox-bridge][ERROR] Invalid steering stick: ${stick}`);
        printHelpAndExit(1);
      }
    } else if (arg === "--hold-ramp-delay-s") {
      cfg.holdForwardRampDelayS = Number(next() || cfg.holdForwardRampDelayS);
    } else if (arg === "--hold-ramp-duration-s") {
      cfg.holdForwardRampDurationS = Number(next() || cfg.holdForwardRampDurationS);
    } else if (arg === "--session-toggle-trigger-threshold") {
      cfg.sessionToggleTriggerThreshold = Number(next() || cfg.sessionToggleTriggerThreshold);
    } else if (arg === "--require-a-button") {
      cfg.requireAButton = true;
    } else if (arg === "--help" || arg === "-h") {
      printHelpAndExit(0);
    } else {
      console.error(`[APEX][xbox-bridge][ERROR] Unknown argument: ${arg}`);
      printHelpAndExit(1);
    }
  }
  cfg.baseLinearSpeedMps = Math.max(0.05, Number(cfg.baseLinearSpeedMps));
  cfg.maxLinearSpeedMps = Math.max(cfg.baseLinearSpeedMps, Number(cfg.maxLinearSpeedMps));
  cfg.holdForwardRampDelayS = Math.max(0.0, Number(cfg.holdForwardRampDelayS));
  cfg.holdForwardRampDurationS = Math.max(0.05, Number(cfg.holdForwardRampDurationS));
  cfg.holdForwardActivationAxis = clamp(Number(cfg.holdForwardActivationAxis), 0.05, 0.95);
  cfg.sessionToggleTriggerThreshold = clamp(Number(cfg.sessionToggleTriggerThreshold), 0.05, 0.99);
  return cfg;
}

function applyDeadband(value, deadband) {
  if (Math.abs(value) <= deadband) {
    return 0.0;
  }
  const scaled = (Math.abs(value) - deadband) / Math.max(1.0e-6, 1.0 - deadband);
  return Math.sign(value) * Math.min(1.0, scaled);
}

function clamp(value, minValue, maxValue) {
  return Math.max(minValue, Math.min(maxValue, value));
}

function buildPowerShellScript(pollMs) {
  return `
$ErrorActionPreference = "Stop"
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public static class XInputBridge {
  [StructLayout(LayoutKind.Sequential)]
  public struct XINPUT_GAMEPAD {
    public ushort wButtons;
    public byte bLeftTrigger;
    public byte bRightTrigger;
    public short sThumbLX;
    public short sThumbLY;
    public short sThumbRX;
    public short sThumbRY;
  }

  [StructLayout(LayoutKind.Sequential)]
  public struct XINPUT_STATE {
    public uint dwPacketNumber;
    public XINPUT_GAMEPAD Gamepad;
  }

  [DllImport("xinput1_4.dll", EntryPoint = "XInputGetState")]
  private static extern uint XInputGetState14(uint dwUserIndex, out XINPUT_STATE pState);

  [DllImport("xinput1_3.dll", EntryPoint = "XInputGetState")]
  private static extern uint XInputGetState13(uint dwUserIndex, out XINPUT_STATE pState);

  public static uint GetState(uint dwUserIndex, out XINPUT_STATE pState) {
    try { return XInputGetState14(dwUserIndex, out pState); }
    catch (DllNotFoundException) { return XInputGetState13(dwUserIndex, out pState); }
  }
}
"@

function NormalizeThumb([int]$value, [int]$deadzone) {
  if ([math]::Abs($value) -le $deadzone) { return 0.0 }
  $sign = if ($value -ge 0) { 1.0 } else { -1.0 }
  $scaled = ([math]::Abs($value) - $deadzone) / (32767.0 - $deadzone)
  if ($scaled -gt 1.0) { $scaled = 1.0 }
  return $sign * $scaled
}

while ($true) {
  $found = $false
  $slot = -1
  $state = New-Object XInputBridge+XINPUT_STATE
  for ($i = 0; $i -lt 4; $i++) {
    $result = [XInputBridge]::GetState([uint32]$i, [ref]$state)
    if ($result -eq 0) {
      $found = $true
      $slot = $i
      break
    }
  }

  if (-not $found) {
    [Console]::WriteLine('{"controller_connected":false,"enabled":false}')
    Start-Sleep -Milliseconds ${pollMs}
    continue
  }

  $buttons = [int]$state.Gamepad.wButtons
  $payload = [ordered]@{
    controller_connected = $true
    slot = $slot
    left_x = [math]::Round((NormalizeThumb $state.Gamepad.sThumbLX 7849), 6)
    left_y = [math]::Round((NormalizeThumb $state.Gamepad.sThumbLY 7849), 6)
    right_x = [math]::Round((NormalizeThumb $state.Gamepad.sThumbRX 8689), 6)
    right_y = [math]::Round((NormalizeThumb $state.Gamepad.sThumbRY 8689), 6)
    left_trigger = [math]::Round(([double]$state.Gamepad.bLeftTrigger / 255.0), 6)
    right_trigger = [math]::Round(([double]$state.Gamepad.bRightTrigger / 255.0), 6)
    a = (($buttons -band 0x1000) -ne 0)
    b = (($buttons -band 0x2000) -ne 0)
    x = (($buttons -band 0x4000) -ne 0)
    y = (($buttons -band 0x8000) -ne 0)
    start = (($buttons -band 0x0010) -ne 0)
    back = (($buttons -band 0x0020) -ne 0)
    packet = [int64]$state.dwPacketNumber
    stamp_ms = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
    device_name = "Xbox Controller"
  }
  [Console]::WriteLine(($payload | ConvertTo-Json -Compress))
  Start-Sleep -Milliseconds ${pollMs}
}
`;
}

function createTempPowerShellScript(pollMs) {
  const tempPath = path.join(os.tmpdir(), "apex_xbox_bridge_reader.ps1");
  fs.writeFileSync(tempPath, buildPowerShellScript(pollMs), "utf8");
  return tempPath;
}

function computeCommand(rawState, cfg, rampState) {
  const controllerConnected = !!rawState.controller_connected;
  const rawLinearAxis = controllerConnected
    ? clamp(applyDeadband(Number(rawState.left_y || 0.0), cfg.axisDeadband), -1.0, 1.0)
    : 0.0;
  const steeringValue =
    cfg.steeringStick === "left" ? Number(rawState.left_x || 0.0) : Number(rawState.right_x || 0.0);
  const rawSteeringAxis = controllerConnected
    ? clamp(applyDeadband(steeringValue, cfg.axisDeadband), -1.0, 1.0)
    : 0.0;
  const enabled = controllerConnected && (!cfg.requireAButton || !!rawState.a);
  const nowMs = Number(rawState.stamp_ms || Date.now());
  const leftTrigger = controllerConnected ? clamp(Number(rawState.left_trigger || 0.0), 0.0, 1.0) : 0.0;
  const rightTrigger = controllerConnected ? clamp(Number(rawState.right_trigger || 0.0), 0.0, 1.0) : 0.0;
  const sessionTogglePressed =
    controllerConnected &&
    leftTrigger >= cfg.sessionToggleTriggerThreshold &&
    rightTrigger >= cfg.sessionToggleTriggerThreshold;

  if (enabled && rawLinearAxis >= cfg.holdForwardActivationAxis) {
    if (rampState.forwardHoldStartMs === null) {
      rampState.forwardHoldStartMs = nowMs;
    }
  } else {
    rampState.forwardHoldStartMs = null;
  }

  let currentLinearCapMps = cfg.baseLinearSpeedMps;
  if (rampState.forwardHoldStartMs !== null) {
    const heldForS = Math.max(0.0, (nowMs - rampState.forwardHoldStartMs) / 1000.0);
    if (heldForS > cfg.holdForwardRampDelayS) {
      const progress = Math.min(
        1.0,
        (heldForS - cfg.holdForwardRampDelayS) / cfg.holdForwardRampDurationS
      );
      currentLinearCapMps =
        cfg.baseLinearSpeedMps +
        progress * (cfg.maxLinearSpeedMps - cfg.baseLinearSpeedMps);
    }
  }

  const linearCapMps = rawLinearAxis >= 0.0 ? currentLinearCapMps : cfg.baseLinearSpeedMps;
  const linearX = enabled ? rawLinearAxis * linearCapMps : 0.0;
  const steeringDeg = enabled ? rawSteeringAxis * cfg.maxSteeringDeg : 0.0;
  const angularZ =
    Math.abs(linearX) > 1.0e-4
      ? (linearX * Math.tan((steeringDeg * Math.PI) / 180.0)) / cfg.wheelbaseM
      : 0.0;
  return {
    controller_connected: controllerConnected,
    enabled,
    device_name: rawState.device_name || "Xbox Controller",
    start_pressed: sessionTogglePressed,
    left_trigger: leftTrigger,
    right_trigger: rightTrigger,
    start_button: !!rawState.start,
    raw_linear_axis: rawLinearAxis,
    raw_steering_axis: rawSteeringAxis,
    steering_source: cfg.steeringStick,
    current_linear_cap_mps: currentLinearCapMps,
    linear_x_mps: linearX,
    angular_z_rps: angularZ,
    steering_deg: steeringDeg,
    stamp_ms: nowMs,
  };
}

function main() {
  const cfg = parseArgs(process.argv.slice(2));
  const tempPs1 = createTempPowerShellScript(cfg.pollMs);
  const rampState = { forwardHoldStartMs: null };
  let latestPayload = {
    controller_connected: false,
    enabled: false,
    current_linear_cap_mps: cfg.baseLinearSpeedMps,
    linear_x_mps: 0.0,
    angular_z_rps: 0.0,
    steering_deg: 0.0,
    stamp_ms: Date.now(),
  };
  let socket = null;
  let connected = false;
  let reconnectTimer = null;

  function clearReconnect() {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
  }

  function scheduleReconnect() {
    if (reconnectTimer) {
      return;
    }
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      connectSocket();
    }, 1000);
  }

  function sendLatest() {
    if (socket && connected && socket.writable) {
      socket.write(`${JSON.stringify(latestPayload)}\n`);
    }
  }

  function connectSocket() {
    if (socket) {
      socket.removeAllListeners();
      socket.destroy();
      socket = null;
      connected = false;
    }
    clearReconnect();
    socket = new net.Socket();
    socket.setNoDelay(true);
    socket.on("connect", () => {
      connected = true;
      console.log(`[APEX][xbox-bridge] connected to ${cfg.host}:${cfg.port}`);
      sendLatest();
    });
    socket.on("error", (error) => {
      if (connected) {
        console.error(`[APEX][xbox-bridge][ERROR] socket: ${error.message}`);
      }
    });
    socket.on("close", () => {
      if (connected) {
        console.log("[APEX][xbox-bridge] bridge socket closed; retrying");
      }
      connected = false;
      scheduleReconnect();
    });
    socket.connect(cfg.port, cfg.host);
  }

  const ps = childProcess.spawn(
    "powershell.exe",
    ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", tempPs1],
    {
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
    }
  );

  ps.stderr.on("data", (chunk) => {
    const text = chunk.toString("utf8").trim();
    if (text) {
      console.error(`[APEX][xbox-bridge][powershell] ${text}`);
    }
  });

  ps.on("exit", (code) => {
    console.error(`[APEX][xbox-bridge][ERROR] controller reader exited with code ${code}`);
    process.exit(code || 1);
  });

  const rl = readline.createInterface({ input: ps.stdout });
  rl.on("line", (line) => {
    const trimmed = line.trim();
    if (!trimmed) {
      return;
    }
    try {
      const rawState = JSON.parse(trimmed);
      latestPayload = computeCommand(rawState, cfg, rampState);
      sendLatest();
    } catch (error) {
      console.error(`[APEX][xbox-bridge][ERROR] bad reader line: ${error.message}`);
    }
  });

  connectSocket();

  console.log("[APEX][xbox-bridge] started");
  console.log(
    `[APEX][xbox-bridge] mapping: left stick Y -> speed, ${cfg.steeringStick} stick X -> steering`
  );
  console.log(
    `[APEX][xbox-bridge] session toggle: both triggers >= ${cfg.sessionToggleTriggerThreshold.toFixed(2)}`
  );
  console.log(
    `[APEX][xbox-bridge] forward hold boost: ${cfg.baseLinearSpeedMps.toFixed(2)} -> ` +
      `${cfg.maxLinearSpeedMps.toFixed(2)} m/s after ${cfg.holdForwardRampDelayS.toFixed(2)} s`
  );
  console.log(
    `[APEX][xbox-bridge] target WSL bridge: ${cfg.host}:${cfg.port} ` +
      `(requireA=${cfg.requireAButton ? "yes" : "no"})`
  );

  function shutdown() {
    latestPayload = {
      controller_connected: false,
      enabled: false,
      current_linear_cap_mps: cfg.baseLinearSpeedMps,
      linear_x_mps: 0.0,
      angular_z_rps: 0.0,
      steering_deg: 0.0,
      stamp_ms: Date.now(),
    };
    try {
      sendLatest();
    } catch (error) {
      // no-op on shutdown
    }
    try {
      rl.close();
    } catch (error) {
      // no-op on shutdown
    }
    try {
      ps.kill();
    } catch (error) {
      // no-op on shutdown
    }
    try {
      clearReconnect();
      if (socket) {
        socket.end();
        socket.destroy();
      }
    } catch (error) {
      // no-op on shutdown
    }
    process.exit(0);
  }

  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

main();
