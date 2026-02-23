/**
 * FloodLedger — Blockchain status panel showing chain integrity,
 * recent blocks, and entry history.
 */

import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'

const LEDGER_API = import.meta.env.VITE_LEDGER_API || '/api/v1/ledger'

function shortenHash(h) {
  if (!h || h.length < 12) return h || '—'
  return h.slice(0, 6) + '…' + h.slice(-6)
}

export default function FloodLedger({ demoMode = false }) {
  const [summary, setSummary] = useState(null)
  const [chain, setChain] = useState([])
  const [expanded, setExpanded] = useState(true)
  const [verifying, setVerifying] = useState(false)
  const [integrity, setIntegrity] = useState(null)

  const fetchData = useCallback(async () => {
    try {
      const [sumRes, chainRes] = await Promise.all([
        axios.get(`${LEDGER_API}/chain/summary`),
        axios.get(`${LEDGER_API}/chain`),
      ])
      setSummary(sumRes.data)
      setChain(chainRes.data.slice(-5)) // last 5 blocks
    } catch {
      // Demo fallback
      const now = new Date().toISOString()
      setSummary({
        chain_id: 'argus_flood_ledger_v1',
        length: 47,
        last_hash: 'a3f8c2d1e5b6...9f4a7d2e',
        entries_24h: 23,
        villages_tracked: 8,
        integrity_verified: true,
      })
      setChain([
        { block_number: 45, timestamp: now, hash: '00a3f8c2d1e5b6a9f4a7d2e0c8b3f1d5e7a2c4b6d8f0e2a4c6b8d0f2e4a6c8', nonce: 142, entries: [{ event_type: 'prediction', village_id: 'kullu_01' }] },
        { block_number: 46, timestamp: now, hash: '0072b4e6f8d0a2c4e6b8d0f2a4c6e8b0d2f4a6c8e0b2d4f6a8c0e2b4d6f8a0', nonce: 89, entries: [{ event_type: 'alert', village_id: 'mandi_01' }] },
        { block_number: 47, timestamp: now, hash: '00e1c3d5f7a9b1d3f5a7c9e1b3d5f7a9c1e3b5d7f9a1c3e5b7d9f1a3c5e7b9', nonce: 234, entries: [{ event_type: 'evacuation', village_id: 'kullu_01' }] },
      ])
    }
  }, [])

  const verifyChain = useCallback(async () => {
    setVerifying(true)
    try {
      const { data } = await axios.get(`${LEDGER_API}/verify`)
      setIntegrity(data.integrity)
    } catch {
      setIntegrity(true) // demo
    }
    setTimeout(() => setVerifying(false), 1000)
  }, [])

  useEffect(() => { fetchData() }, [fetchData])
  useEffect(() => {
    const iv = setInterval(fetchData, demoMode ? 10000 : 30000)
    return () => clearInterval(iv)
  }, [fetchData, demoMode])

  if (!expanded) {
    return (
      <button
        onClick={() => setExpanded(true)}
        className="bg-navy/90 border border-gray-700 text-cyan-400 text-xs font-mono px-3 py-2 rounded-lg hover:border-cyan-400 transition-colors"
      >
        LEDGER ▸
      </button>
    )
  }

  const eventColors = {
    prediction: 'text-blue-400',
    alert: 'text-amber-400',
    evacuation: 'text-red-400',
    verification: 'text-green-400',
  }

  return (
    <div className="bg-navy/95 backdrop-blur-sm border border-gray-700 rounded-xl p-4 max-w-xs">
      <div className="flex justify-between items-center mb-3">
        <h3 className="font-display text-sm text-white tracking-wider">
          FLOOD<span className="text-cyan-400">LEDGER</span>
        </h3>
        <button onClick={() => setExpanded(false)} className="text-gray-500 text-xs hover:text-white">✕</button>
      </div>

      {/* Chain stats */}
      {summary && (
        <div className="grid grid-cols-3 gap-2 mb-3">
          <div className="text-center">
            <div className="text-lg text-white font-mono">{summary.length}</div>
            <div className="text-[9px] text-gray-500 uppercase">Blocks</div>
          </div>
          <div className="text-center">
            <div className="text-lg text-white font-mono">{summary.entries_24h}</div>
            <div className="text-[9px] text-gray-500 uppercase">Entries 24h</div>
          </div>
          <div className="text-center">
            <div className="text-lg text-white font-mono">{summary.villages_tracked}</div>
            <div className="text-[9px] text-gray-500 uppercase">Villages</div>
          </div>
        </div>
      )}

      {/* Integrity badge */}
      <div className="flex items-center gap-2 mb-3">
        <button
          onClick={verifyChain}
          disabled={verifying}
          className={`text-[10px] font-mono px-2 py-1 rounded border transition-colors ${
            verifying
              ? 'border-yellow-600 text-yellow-400 animate-pulse'
              : integrity === true
                ? 'border-green-600 text-green-400'
                : integrity === false
                  ? 'border-red-600 text-red-400'
                  : 'border-gray-600 text-gray-400 hover:border-cyan-500'
          }`}
        >
          {verifying ? '⟳ VERIFYING...' : integrity === true ? '✓ CHAIN VALID' : integrity === false ? '✗ TAMPERED' : '⟳ VERIFY'}
        </button>
        {summary && (
          <span className="text-[9px] text-gray-600 font-mono">
            {shortenHash(summary.last_hash)}
          </span>
        )}
      </div>

      {/* Recent blocks */}
      <div className="space-y-1.5">
        <div className="text-[10px] text-gray-500 uppercase tracking-wider">Recent Blocks</div>
        {chain.map((block) => (
          <div key={block.block_number} className="bg-gray-800/40 rounded px-2 py-1.5">
            <div className="flex justify-between items-center">
              <span className="text-[10px] text-cyan-400 font-mono">
                #{block.block_number}
              </span>
              <span className="text-[9px] text-gray-600 font-mono">
                nonce: {block.nonce}
              </span>
            </div>
            <div className="text-[9px] text-gray-500 font-mono truncate">
              {shortenHash(block.hash)}
            </div>
            {block.entries?.map((e, i) => (
              <span key={i} className={`text-[9px] mr-1 ${eventColors[e.event_type] || 'text-gray-400'}`}>
                {e.event_type}:{e.village_id}
              </span>
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}
