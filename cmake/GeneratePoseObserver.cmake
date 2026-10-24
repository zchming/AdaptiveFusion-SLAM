# Generate a build-local observer; the production tracking source stays intact.
file(READ "${AFS_PRODUCTION_SOURCE}" _afs_entry)
set(_afs_start [=[        for (std::size_t index = 0; index < frame_count; ++index) {]=])
set(_afs_finish [=[            std::cerr << "Frame " << index << ": "]=])
foreach(_afs_anchor IN ITEMS _afs_start _afs_finish)
    string(FIND "${_afs_entry}" "${${_afs_anchor}}" _afs_position)
    if(_afs_position EQUAL -1)
        message(FATAL_ERROR "Production observer hook changed: ${_afs_anchor}")
    endif()
endforeach()
string(REPLACE "${_afs_start}"
    "${_afs_start}\n            afs_live::writer->wait(dataset.associations()[index]);"
    _afs_entry "${_afs_entry}")
set(_afs_publish [=[            afs_live::writer->publish(result.frame,insert_keyframe,
                insert_keyframe ? static_cast<long long>(sparse_map.keyframes().size()-1) : -1,
                frame_start,frame_end,frame_times_ms.back());
]=])
string(REPLACE "${_afs_finish}" "${_afs_publish}${_afs_finish}" _afs_entry "${_afs_entry}")
file(WRITE "${AFS_GENERATED_SOURCE}" "${_afs_entry}")
