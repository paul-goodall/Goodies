$(function() {
// ================ JS BELOW HERE ================

// ----
function set_the_class(my_selector,my_class){
  $(my_selector).attr('class') = '';
  $(my_selector).addClass(my_class);
}
// ----
// Define the sizes:
function determine_layout(winX,lx,rx){
  var main_width;
  var side_width;
  var frac;
  var main_obj = $('.container_content');
  var filler_objs = $('.container_filler');
  var w1 = Number(main_obj.attr('w1'));
  var w2 = Number(main_obj.attr('w2'));
  var s1 = Number(main_obj.attr('s1'));
  var s2 = Number(main_obj.attr('s2'));
  var numItems = $('.container_filler').length;

  frac = (winX - s1)/(s2 - s1);
  if(frac > 1) frac = 1;
  if(frac < 0) frac = 0;
  main_width = w1 + frac*(w2-w1);
  side_width = Math.floor(100 * ((100 - main_width)/numItems)) / 100;
  main_width = 100 - numItems*side_width;
  main_obj.css('width', main_width + '%');
  filler_objs.css('width', side_width + '%');
}
// ----
function adjustLayout(){
  var winX  = window.innerWidth;
  var lblock = $('#leftblock').width();
  var rblock = $('#rightblock').width();

  determine_layout(winX,lblock,rblock);
}
// ----
adjustLayout();

$('#card_layout_slider').on('input',function(e){
 var view_choice = $(this).val();
 var orig_class;

  if(view_choice == 0){
      var obj = $('div#cardholder');
      orig_class = obj.attr('class');
      obj.removeClass(orig_class);
      obj.addClass('smallcard');
      var obj_span = $('div#cardholder span');
      obj_span.css('height','140px');
  }

  if(view_choice == 1){
    var obj = $('div#cardholder');
    orig_class = obj.attr('class');
    obj.removeClass(orig_class);
    obj.addClass('mediumcard');
    var obj_span = $('div#cardholder span');
    obj_span.css('height','180px');
  }

  if(view_choice == 2){
    var obj = $('div#cardholder');
    orig_class = obj.attr('class');
    obj.removeClass(orig_class);
    obj.addClass('largecard');
    var obj_span = $('div#cardholder span');
    obj_span.css('height','240px');
  }
});


// ----
window.addEventListener("resize", adjustLayout);
// ================ JS ABOVE HERE ================
});
