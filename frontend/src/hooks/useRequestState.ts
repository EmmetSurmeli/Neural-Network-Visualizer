import {useCallback, useEffect, useState} from 'react';
import {ApiError, readableError} from '../lib/http';
export type RequestStatus = 'idle' | 'loading' | 'success' | 'empty' | 'error' | 'stale' | 'offline';
export interface RequestState {status: RequestStatus; message?: string; error?: ApiError; previous?: boolean}
export default function useRequestState() {
  const [state, setState] = useState<RequestState>({status: 'idle'});
  const start = useCallback((message: string) => setState({status: 'loading', message}), []);
  const success = useCallback((empty = false) => setState({status: empty ? 'empty' : 'success'}), []);
  const fail = useCallback((value: unknown, previous = false) => {const error = readableError(value); setState({status: error.code === 'OFFLINE' ? 'offline' : previous ? 'stale' : 'error', error, previous});}, []);
  const reset = useCallback(() => setState({status: 'idle'}), []);
  return {state, start, success, fail, reset};
}
export function useOnline() {
  const [online, setOnline] = useState(navigator.onLine);
  // The shell handles the visible offline notice; consumers disable live controls.
  useEffect(() => {const update = () => setOnline(navigator.onLine); window.addEventListener('online', update); window.addEventListener('offline', update); return () => {window.removeEventListener('online', update); window.removeEventListener('offline', update);};}, []);
  return online;
}
