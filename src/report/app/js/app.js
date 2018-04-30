'use strict';

function copy_properties(source, target) {
    for (var prop in source) {
        if (source.hasOwnProperty(prop)) {
            if (source[prop] != null) {
                target[prop] = source[prop];
            }
        }
    }
}

function Dummy(obj) { copy_properties(obj, this); }
function ReportEntry(obj) { copy_properties(obj, this); }

var yaml_type_map = [
    ['!grond.ReportEntry', Dummy],
    ['!grond.ParameterStats', Dummy],
    ['!grond.TargetBalancingAnalyserResult', Dummy],
    ['!grond.ResultStats', Dummy],
    ['!grond.WaveformMisfitTarget', Dummy],
    ['!grond.WaveformMisfitConfig', Dummy],
    ['!grond.WaveformTargetGroup', Dummy],
    ['!grond.PhaseRatioTarget', Dummy],
    ['!grond.PhaseRatioTargetGroup', Dummy],
    ['!grond.FeatureMeasure', Dummy],
    ['!grond.CMTProblem', Dummy],
    ['!pf.MTSource', Dummy],
    ['!pf.HalfSinusoidSTF', Dummy],
    ['!grond.PlotCollection', Dummy],
    ['!grond.PlotGroup', Dummy],
    ['!grond.PlotItem', Dummy],
    ['!grond.PNG', Dummy],
    ['!grond.PDF', Dummy],
];

function make_constructor(type) {
    var type = type;
    var construct = function(data) {
        return new type(data);
    };
    return construct;
}

var yaml_types = [];
for (var i=0; i<yaml_type_map.length; i++) {
    var type = yaml_type_map[i][1]
    var t = new jsyaml.Type(yaml_type_map[i][0], {
        kind: 'mapping',
        instanceOf: type,
        construct: make_constructor(type),
    });
    yaml_types.push(t);
}

var report_schema = jsyaml.Schema.create(yaml_types);

function parse_fields_float(fields, input, output, error, factor) {
    parse_fields(fields, input, output, error, factor, parseFloat);
}

function parse_fields_int(fields, input, output, error) {
    parse_fields(fields, input, output, error, 1.0, parseInt);
}

function parse_fields(fields, input, output, error, factor, parse) {
    for (var i=0; i<fields.length; i++) {
        var field = fields[i];
        if (input[field].length == 0) {
            val = null;
        } else {
            var val = parse(input[field]) * factor;
            if (val.isNaN) {
                error[field] = true;
                return false;
            }
        }
        output[field] = val;
    }
}


angular.module('reportApp', ['ngRoute'])

    .config(function($routeProvider, $locationProvider) {
        $locationProvider.hashPrefix('');
        $routeProvider
            .when('/reports/', {
                controller: 'ReportListController',
                templateUrl: 'report_list.html',
            })
            .when('/reports/:report_path*/', {
                controller: 'ReportController',
                templateUrl:'report.html',
            })
            .otherwise({
                redirectTo: '/reports/',
            });
    })

    .factory('YamlDoc', function($http) {

        var funcs = {};
        funcs.query = function(path, loaded, options) {
            $http.get(path).then(
                function(response) {
                    var doc = jsyaml.safeLoad(response.data, options);
                    loaded(doc);
                }
            );
        };
        return funcs;
    })

    .factory('YamlMultiDoc', function($http) {

        var funcs = {};
        funcs.query = function(path, loaded, options) {
            $http.get(path).then(
                function(response) {
                    jsyaml.safeLoadAll(response.data, loaded, options);
                }
            );
        };
        return funcs;
    })

    .controller('NavigationController', function($scope, $route, YamlDoc, YamlMultiDoc, $routeParams, $location) {
        $scope.$route = $route;
        $scope.$location = $location;
        $scope.$routeParams = $routeParams;

        $scope.active = function(path) {
            return (path === $location.path().substr(0,path.length)) ? 'active' : '';
        };

    })

    .controller('ReportListController', function($scope, YamlMultiDoc) {
        $scope.report_entries = [];

        YamlMultiDoc.query(
            'report_list.yaml',
            function(doc) { $scope.report_entries.push(doc); console.log(doc); },
            {schema: report_schema});
    })

    .controller('ReportController', function(
            $scope, YamlDoc, YamlMultiDoc, $routeParams, $location, $anchorScroll) {

        $scope.stats = null;
        $scope.plot_groups = [];
        $anchorScroll.yOffset = 60;

        $scope.path = $routeParams.report_path;

        var plot_group_path = function(group_ref) {
            return $scope.path + '/plots/' + group_ref[0] + '/' + group_ref[1] + '/' + group_ref[0] + '.' + group_ref[1];
        };

        $scope.image_path = function(group, item) {
            return plot_group_path([group.name, group.variant]) + '.' + item.name + '.d100.png'
        }

        YamlDoc.query(
            $scope.path + '/stats.yaml',
            function(doc) { $scope.stats = doc; },
            {schema: report_schema});


        var query_group = function(group_ref) {
            YamlDoc.query(
                plot_group_path(group_ref) + '.plot_group.yaml',
                function(doc) { $scope.plot_groups.push(doc); },
                {schema: report_schema});
        };

        YamlMultiDoc.query(
            $scope.path + '/plots/plot_collection.yaml',
            function(doc) {
                Array.forEach(
                    doc.group_refs,
                    query_group);
                },
            {schema: report_schema});

        $scope.scrollTo = function(id) {
            var old = $location.hash();
            $location.hash(id);
            $anchorScroll();
            //reset to old to keep any additional routing logic from kicking in
            $location.hash(old);
        };
    })

    .filter('eround', function() {
        return function(input, std) {
            if (std > 0) {
                var ndig = - Math.floor(Math.log10(std)) + 1;
                var factor = Math.pow(10, ndig);
                return Math.round(input * factor) / factor;
            } else {
                return input;
            }
        };
    })

    .filter('dotalign', function() {
        return function(input) {
            input = input.toString();
            var dotpos = input.indexOf('.');
            if (dotpos == -1) {
                dotpos = input.length;
            }
            var fill = ' ';
            return fill.repeat(Math.max(0, 5 - dotpos)) + input;
        };
    })

    .run(function($rootScope, $location, $anchorScroll, $routeParams) {
      //when the route is changed scroll to the proper element.
      $rootScope.$on('$routeChangeSuccess', function(newRoute, oldRoute) {
        $location.hash($routeParams.scrollTo);
        $anchorScroll();  
      });
    });
