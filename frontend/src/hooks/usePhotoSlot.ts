import { useCallback, useEffect, useRef, useState } from 'react'

import { ApiError, inspectPhoto } from '../api/client'
import type { PhotoInfo } from '../api/types'

/**
 * Everything one upload box knows. Exactly one of four states, so the page
 * can never show, say, a preview and an error for the same photo at once.
 */
export type PhotoSlotState =
  | { status: 'empty' }
  | { status: 'checking'; file: File; objectUrl: string }
  | { status: 'ready'; file: File; objectUrl: string; info: PhotoInfo }
  | { status: 'invalid'; file: File; message: string }

/**
 * Holds one photo and checks it with the server as soon as it is chosen.
 * If a second photo is chosen while the first is still being checked, the
 * first check is cancelled, so a slow answer can never overwrite a newer one.
 */
export function usePhotoSlot() {
  const [state, setState] = useState<PhotoSlotState>({ status: 'empty' })
  const pending = useRef<AbortController | null>(null)
  const objectUrl = useRef<string | null>(null)

  const releaseObjectUrl = () => {
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current)
    objectUrl.current = null
  }

  const choose = useCallback(async (file: File) => {
    pending.current?.abort()
    const controller = new AbortController()
    pending.current = controller
    releaseObjectUrl()
    const url = URL.createObjectURL(file)
    objectUrl.current = url
    setState({ status: 'checking', file, objectUrl: url })
    try {
      const info = await inspectPhoto(file, controller.signal)
      if (!controller.signal.aborted) setState({ status: 'ready', file, objectUrl: url, info })
    } catch (error) {
      if (controller.signal.aborted) return
      const message = error instanceof ApiError ? error.message : 'the photo could not be checked'
      setState({ status: 'invalid', file, message })
    }
  }, [])

  useEffect(
    () => () => {
      pending.current?.abort()
      releaseObjectUrl()
    },
    [],
  )

  return { state, choose }
}
