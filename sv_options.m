function o = sv_options(defaults, args)
%SV_OPTIONS  Merge name/value pairs (or a struct) into a defaults struct.

o = defaults;
if isempty(args), return; end

if numel(args) == 1 && isstruct(args{1})
    s = args{1};
    f = fieldnames(s);
    for i = 1:numel(f)
        o = setopt(o, f{i}, s.(f{i}));
    end
    return
end

if mod(numel(args), 2) ~= 0
    error('sv_options:pairs', 'options must be given as name/value pairs.');
end
for i = 1:2:numel(args)
    o = setopt(o, args{i}, args{i+1});
end
end

function o = setopt(o, name, value)
if ~ischar(name)
    error('sv_options:name', 'option names must be strings.');
end
f = fieldnames(o);
hit = find(strcmpi(f, name), 1);
if isempty(hit)
    error('sv_options:unknown', 'unknown option ''%s''.', name);
end
o.(f{hit}) = value;
end
