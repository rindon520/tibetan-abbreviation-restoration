from pyplusplus import module_builder 

mb = module_builder.module_builder_t(
    files=['Segmentor.cpp',
        'ChType.cpp',
        'ChType.h',
        'DataStruct.h',
        'Perceptron.h',
        'Perceptron.cpp',
        'Segmentor.h',
        'splitutf.h',
        'splitutf.cpp',
        'Utf8Stream.cpp',
        'Utf8Stream.h'],
    xml_generator_path='/usr/bin/castxml'
)
    #  files=['Segmentor.cpp',
    #         'Segmentor.h',
    #     ],xml_generator_path='/usr/bin/castxml'
# )

mb.build_code_creator(module_name='tiseg')

mb.write_module('tiseg.cc')

