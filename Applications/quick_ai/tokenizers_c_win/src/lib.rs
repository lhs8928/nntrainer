use libc::{c_char, c_int, c_void, size_t};
use std::ptr;
use std::slice;
use tokenizers::models::bpe::{Vocab, BPE};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Tokenizer};

type TokenizerHandle = *mut c_void;

#[repr(C)]
pub struct TokenizerEncodeResult {
    token_ids: *mut c_int,
    len: size_t,
}

struct TokenizerState {
    tokenizer: Tokenizer,
    decoded: Vec<u8>,
    token: Vec<u8>,
}

unsafe fn bytes_from_raw<'a>(data: *const c_char, len: size_t) -> Option<&'a [u8]> {
    if data.is_null() {
        return if len == 0 { Some(&[]) } else { None };
    }
    Some(slice::from_raw_parts(data as *const u8, len))
}

unsafe fn state_from_handle<'a>(handle: TokenizerHandle) -> Option<&'a mut TokenizerState> {
    if handle.is_null() {
        return None;
    }
    Some(&mut *(handle as *mut TokenizerState))
}

unsafe fn fill_result(result: *mut TokenizerEncodeResult, ids: &[u32]) {
    if result.is_null() {
        return;
    }

    if ids.is_empty() {
        (*result).token_ids = ptr::null_mut();
        (*result).len = 0;
        return;
    }

    let byte_len = ids.len() * std::mem::size_of::<c_int>();
    let out = libc::malloc(byte_len) as *mut c_int;
    if out.is_null() {
        (*result).token_ids = ptr::null_mut();
        (*result).len = 0;
        return;
    }

    for (idx, id) in ids.iter().enumerate() {
        *out.add(idx) = *id as c_int;
    }

    (*result).token_ids = out;
    (*result).len = ids.len();
}

fn parse_merges(data: &[u8]) -> Option<Vec<(String, String)>> {
    let text = std::str::from_utf8(data).ok()?;
    let mut merges = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("#version") {
            continue;
        }

        let mut parts = line.split(' ');
        let first = parts.next()?;
        let second = parts.next()?;
        if parts.next().is_some() {
            return None;
        }

        merges.push((first.to_string(), second.to_string()));
    }

    Some(merges)
}

fn parse_added_tokens(data: &[u8]) -> Option<Vec<AddedToken>> {
    if data.is_empty() {
        return Some(Vec::new());
    }

    if let Ok(tokens) = serde_json::from_slice::<Vec<AddedToken>>(data) {
        return Some(tokens);
    }

    let tokens = serde_json::from_slice::<Vec<String>>(data).ok()?;
    Some(
        tokens
            .into_iter()
            .map(|token| AddedToken::from(token, false))
            .collect(),
    )
}

#[no_mangle]
pub unsafe extern "C" fn tokenizers_new_from_str(
    json: *const c_char,
    len: size_t,
) -> TokenizerHandle {
    let Some(bytes) = bytes_from_raw(json, len) else {
        return ptr::null_mut();
    };

    match Tokenizer::from_bytes(bytes) {
        Ok(tokenizer) => Box::into_raw(Box::new(TokenizerState {
            tokenizer,
            decoded: Vec::new(),
            token: Vec::new(),
        })) as TokenizerHandle,
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn byte_level_bpe_tokenizers_new_from_str(
    vocab: *const c_char,
    vocab_len: size_t,
    merges: *const c_char,
    merges_len: size_t,
    added_tokens: *const c_char,
    added_tokens_len: size_t,
) -> TokenizerHandle {
    let Some(vocab_bytes) = bytes_from_raw(vocab, vocab_len) else {
        return ptr::null_mut();
    };
    let Some(merges_bytes) = bytes_from_raw(merges, merges_len) else {
        return ptr::null_mut();
    };
    let Some(added_tokens_bytes) = bytes_from_raw(added_tokens, added_tokens_len) else {
        return ptr::null_mut();
    };
    let Ok(vocab) = serde_json::from_slice::<Vocab>(vocab_bytes) else {
        return ptr::null_mut();
    };
    let Some(merges) = parse_merges(merges_bytes) else {
        return ptr::null_mut();
    };
    let Some(added_tokens) = parse_added_tokens(added_tokens_bytes) else {
        return ptr::null_mut();
    };

    let Ok(model) = BPE::builder().vocab_and_merges(vocab, merges).build() else {
        return ptr::null_mut();
    };

    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));
    tokenizer.with_post_processor(Some(ByteLevel::default()));
    tokenizer.with_decoder(Some(ByteLevel::default()));
    tokenizer.add_tokens(&added_tokens);

    Box::into_raw(Box::new(TokenizerState {
        tokenizer,
        decoded: Vec::new(),
        token: Vec::new(),
    })) as TokenizerHandle
}

#[no_mangle]
pub unsafe extern "C" fn tokenizers_encode(
    handle: TokenizerHandle,
    data: *const c_char,
    len: size_t,
    add_special_token: c_int,
    result: *mut TokenizerEncodeResult,
) {
    let Some(state) = state_from_handle(handle) else {
        fill_result(result, &[]);
        return;
    };
    let Some(bytes) = bytes_from_raw(data, len) else {
        fill_result(result, &[]);
        return;
    };
    let Ok(text) = std::str::from_utf8(bytes) else {
        fill_result(result, &[]);
        return;
    };

    match state.tokenizer.encode(text, add_special_token != 0) {
        Ok(encoding) => fill_result(result, encoding.get_ids()),
        Err(_) => fill_result(result, &[]),
    }
}

#[no_mangle]
pub unsafe extern "C" fn tokenizers_encode_batch(
    handle: TokenizerHandle,
    data: *const *const c_char,
    len: *mut size_t,
    num_seqs: size_t,
    add_special_token: c_int,
    results: *mut TokenizerEncodeResult,
) {
    if data.is_null() || len.is_null() || results.is_null() {
        return;
    }

    for idx in 0..num_seqs {
        tokenizers_encode(
            handle,
            *data.add(idx),
            *len.add(idx),
            add_special_token,
            results.add(idx),
        );
    }
}

#[no_mangle]
pub unsafe extern "C" fn tokenizers_free_encode_results(
    results: *mut TokenizerEncodeResult,
    num_seqs: size_t,
) {
    if results.is_null() {
        return;
    }

    for idx in 0..num_seqs {
        let result = results.add(idx);
        if !(*result).token_ids.is_null() {
            libc::free((*result).token_ids as *mut c_void);
        }
        (*result).token_ids = ptr::null_mut();
        (*result).len = 0;
    }
}

#[no_mangle]
pub unsafe extern "C" fn tokenizers_decode(
    handle: TokenizerHandle,
    data: *const u32,
    len: size_t,
    skip_special_token: c_int,
) {
    let Some(state) = state_from_handle(handle) else {
        return;
    };
    if data.is_null() {
        state.decoded.clear();
        return;
    }

    let ids = slice::from_raw_parts(data, len);
    match state.tokenizer.decode(ids, skip_special_token != 0) {
        Ok(decoded) => state.decoded = decoded.into_bytes(),
        Err(_) => state.decoded.clear(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn tokenizers_get_decode_str(
    handle: TokenizerHandle,
    data: *mut *const c_char,
    len: *mut size_t,
) {
    if data.is_null() || len.is_null() {
        return;
    }
    let Some(state) = state_from_handle(handle) else {
        *data = ptr::null();
        *len = 0;
        return;
    };

    *data = state.decoded.as_ptr() as *const c_char;
    *len = state.decoded.len();
}

#[no_mangle]
pub unsafe extern "C" fn tokenizers_get_vocab_size(handle: TokenizerHandle, size: *mut size_t) {
    if size.is_null() {
        return;
    }
    let Some(state) = state_from_handle(handle) else {
        *size = 0;
        return;
    };

    *size = state.tokenizer.get_vocab_size(true) as size_t;
}

#[no_mangle]
pub unsafe extern "C" fn tokenizers_id_to_token(
    handle: TokenizerHandle,
    id: u32,
    data: *mut *const c_char,
    len: *mut size_t,
) {
    if data.is_null() || len.is_null() {
        return;
    }
    let Some(state) = state_from_handle(handle) else {
        *data = ptr::null();
        *len = 0;
        return;
    };

    state.token = state
        .tokenizer
        .id_to_token(id)
        .unwrap_or_default()
        .into_bytes();
    *data = state.token.as_ptr() as *const c_char;
    *len = state.token.len();
}

#[no_mangle]
pub unsafe extern "C" fn tokenizers_token_to_id(
    handle: TokenizerHandle,
    token: *const c_char,
    len: size_t,
    id: *mut i32,
) {
    if id.is_null() {
        return;
    }
    let Some(state) = state_from_handle(handle) else {
        *id = -1;
        return;
    };
    let Some(bytes) = bytes_from_raw(token, len) else {
        *id = -1;
        return;
    };
    let Ok(text) = std::str::from_utf8(bytes) else {
        *id = -1;
        return;
    };

    *id = state
        .tokenizer
        .token_to_id(text)
        .map(|token_id| token_id as i32)
        .unwrap_or(-1);
}

#[no_mangle]
pub unsafe extern "C" fn tokenizers_free(handle: TokenizerHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle as *mut TokenizerState));
    }
}
